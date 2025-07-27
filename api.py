#!/usr/bin/env python3
"""
FastAPI production server for JIVI Cataract Classification System.
Provides REST API endpoints for model inference, training, and monitoring.
Uses Ray for concurrent request processing.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import ray

# Import our system components
from main import JIVICataractSystem
from src.config.config import TrainingConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="JIVI Cataract Classification API",
    description="Production API for automated cataract detection using CLIP and LGBM models with Ray-based concurrent processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global system instance (loaded once at startup)
system: Optional[JIVICataractSystem] = None
config = TrainingConfig()

# Ray configuration
RAY_INITIALIZED = False
RAY_WORKERS = []

# Paths
BASE_PATH = "customer_data"
TRAIN_PATH = "training_data"
MODELS_PATH = "models"

# Ensure directories exist
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    processing_time: float
    file_path: str
    timestamp: str
    ray_worker_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_images: int
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    processing_time: float
    model_used: str
    timestamp: str
    ray_workers_used: int


class TrainingResponse(BaseModel):
    status: str
    model: str
    training_time: float
    metrics: Optional[Dict[str, Any]] = None
    message: str


class DriftResponse(BaseModel):
    unlabeled_samples: int
    drift_score: float
    last_updated: str
    model_performance: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    system_ready: bool
    device: str
    timestamp: str
    ray_initialized: bool
    ray_workers_available: int


# Ray remote functions for concurrent processing
@ray.remote
class PredictionWorker:
    """Ray worker for handling prediction requests."""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'cpu'):
        self.model_type = model_type
        self.device = device
        self.model_path = model_path
        self.inference = None
        self._load_model()
        
    def _load_model(self):
        """Load the appropriate model."""
        try:
            if self.model_type == 'clip':
                from src.inference.clip_inference import CLIPInference
                self.inference = CLIPInference(self.model_path, self.device)
            elif self.model_type == 'lgbm':
                from src.inference.lgbm_inference import LGBMInference
                self.inference = LGBMInference(self.model_path, self.device)
            logger.info(f"Ray worker loaded {self.model_type} model")
        except Exception as e:
            logger.error(f"Failed to load model in Ray worker: {str(e)}")
            raise
            
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """Predict on a single image."""
        try:
            result = self.inference.predict_single(image_path)
            result['ray_worker_id'] = ray.get_runtime_context().get_worker_id()
            return result
        except Exception as e:
            logger.error(f"Prediction error in Ray worker: {str(e)}")
            raise
            
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict on a batch of images."""
        try:
            results = self.inference.predict_batch(image_paths)
            for result in results:
                result['ray_worker_id'] = ray.get_runtime_context().get_worker_id()
            return results
        except Exception as e:
            logger.error(f"Batch prediction error in Ray worker: {str(e)}")
            raise


def initialize_ray():
    """Initialize Ray for distributed processing."""
    global RAY_INITIALIZED, RAY_WORKERS
    
    try:
        if not ray.is_initialized():
            ray.init(
                num_cpus=4,  # Adjust based on your system
                ignore_reinit_error=True,
                log_to_driver=False
            )
            logger.info("Ray initialized successfully")
        
        RAY_INITIALIZED = True
        
        # Create Ray workers for different models
        clip_worker = PredictionWorker.remote(
            model_path="models/clip/best_model.pth",
            model_type="clip",
            device="cpu"
        )
        
        lgbm_worker = PredictionWorker.remote(
            model_path="models/lgbm/trained_model.pkl",
            model_type="lgbm",
            device="cpu"
        )
        
        RAY_WORKERS = {
            'clip': clip_worker,
            'lgbm': lgbm_worker
        }
        
        logger.info(f"Created {len(RAY_WORKERS)} Ray workers")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {str(e)}")
        return False


def initialize_system():
    """Initialize the JIVI Cataract Classification System."""
    global system
    try:
        logger.info("Initializing JIVI Cataract Classification System...")
        system = JIVICataractSystem(device='auto')
        
        # Load models
        system.load_models(load_clip=True, load_lgbm=True)
        
        # Initialize Ray
        initialize_ray()
        
        logger.info("System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    success = initialize_system()
    if not success:
        logger.error("Failed to initialize system on startup")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global system, RAY_INITIALIZED
    
    if system:
        system.shutdown()
        logger.info("System shutdown completed")
    
    if RAY_INITIALIZED and ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown completed")


def save_uploaded_file(upload_file: UploadFile, base_dir: str) -> str:
    """Save uploaded file and return the path."""
    # Create unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    file_extension = Path(upload_file.filename).suffix
    filename = f"{timestamp}_{unique_id}{file_extension}"
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path


def validate_image_file(filename: str) -> bool:
    """Validate that the uploaded file is an image."""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_extension = Path(filename).suffix.lower()
    return file_extension in allowed_extensions


async def predict_with_ray(image_path: str, model_type: str) -> Dict[str, Any]:
    """Predict using Ray worker."""
    global RAY_WORKERS
    
    if model_type not in RAY_WORKERS:
        raise ValueError(f"Model type {model_type} not available in Ray workers")
    
    # Submit prediction task to Ray worker
    future = RAY_WORKERS[model_type].predict_single.remote(image_path)
    result = await asyncio.get_event_loop().run_in_executor(
        None, ray.get, future
    )
    
    return result


async def predict_batch_with_ray(image_paths: List[str], model_type: str) -> List[Dict[str, Any]]:
    """Predict batch using Ray worker."""
    global RAY_WORKERS
    
    if model_type not in RAY_WORKERS:
        raise ValueError(f"Model type {model_type} not available in Ray workers")
    
    # Submit batch prediction task to Ray worker
    future = RAY_WORKERS[model_type].predict_batch.remote(image_paths)
    results = await asyncio.get_event_loop().run_in_executor(
        None, ray.get, future
    )
    
    return results


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global system, RAY_INITIALIZED, RAY_WORKERS
    
    models_loaded = {
        'clip': system.clip_inference is not None if system else False,
        'lgbm': system.lgbm_inference is not None if system else False,
        'batch': system.batch_inference is not None if system else False
    }
    
    system_ready = all(models_loaded.values()) if system else False
    
    return HealthResponse(
        status="ok" if system_ready else "degraded",
        models_loaded=models_loaded,
        system_ready=system_ready,
        device=system.device if system else "unknown",
        timestamp=datetime.now().isoformat(),
        ray_initialized=RAY_INITIALIZED,
        ray_workers_available=len(RAY_WORKERS)
    )


@app.post("/predict_clip", response_model=PredictionResponse)
async def predict_clip(image: UploadFile = File(...)):
    """Predict using CLIP model with Ray."""
    global system, RAY_INITIALIZED
    
    if not system or not system.clip_inference:
        raise HTTPException(status_code=503, detail="CLIP model not loaded")
    
    if not RAY_INITIALIZED:
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "clip"))
        
        # Perform prediction using Ray
        start_time = time.time()
        result = await predict_with_ray(file_path, 'clip')
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=result['class_name'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_used='clip',
            processing_time=processing_time,
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            ray_worker_id=result.get('ray_worker_id')
        )
        
    except Exception as e:
        logger.error(f"CLIP prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_lgbm", response_model=PredictionResponse)
async def predict_lgbm(image: UploadFile = File(...)):
    """Predict using LGBM model with Ray."""
    global system, RAY_INITIALIZED
    
    if not system or not system.lgbm_inference:
        raise HTTPException(status_code=503, detail="LGBM model not loaded")
    
    if not RAY_INITIALIZED:
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "lgbm"))
        
        # Perform prediction using Ray
        start_time = time.time()
        result = await predict_with_ray(file_path, 'lgbm')
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=result['class_name'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_used='lgbm',
            processing_time=processing_time,
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            ray_worker_id=result.get('ray_worker_id')
        )
        
    except Exception as e:
        logger.error(f"LGBM prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_ensemble", response_model=PredictionResponse)
async def predict_ensemble(image: UploadFile = File(...)):
    """Predict using ensemble model (CLIP + LGBM) with Ray."""
    global system, RAY_INITIALIZED
    
    if not system or not system.clip_inference or not system.lgbm_inference:
        raise HTTPException(status_code=503, detail="Ensemble models not loaded")
    
    if not RAY_INITIALIZED:
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "ensemble"))
        
        # Perform ensemble prediction using Ray workers
        start_time = time.time()
        
        # Get predictions from both models concurrently
        clip_future = predict_with_ray(file_path, 'clip')
        lgbm_future = predict_with_ray(file_path, 'lgbm')
        
        clip_result, lgbm_result = await asyncio.gather(clip_future, lgbm_future)
        
        # Weighted ensemble (50-50)
        ensemble_prob_cataract = 0.5 * clip_result['probabilities']['cataract'] + 0.5 * lgbm_result['probabilities']['cataract']
        ensemble_prob_normal = 1 - ensemble_prob_cataract
        
        ensemble_class = 1 if ensemble_prob_cataract > 0.5 else 0
        ensemble_confidence = max(ensemble_prob_normal, ensemble_prob_cataract)
        
        result = {
            'class': ensemble_class,
            'class_name': 'cataract' if ensemble_class == 1 else 'normal',
            'confidence': ensemble_confidence,
            'probabilities': {
                'normal': ensemble_prob_normal,
                'cataract': ensemble_prob_cataract
            },
            'clip_prediction': clip_result,
            'lgbm_prediction': lgbm_result,
            'ray_worker_id': f"{clip_result.get('ray_worker_id')}+{lgbm_result.get('ray_worker_id')}"
        }
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=result['class_name'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_used='ensemble',
            processing_time=processing_time,
            file_path=file_path,
            timestamp=datetime.now().isoformat(),
            ray_worker_id=result.get('ray_worker_id')
        )
        
    except Exception as e:
        logger.error(f"Ensemble prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(
    images: List[UploadFile] = File(...),
    model: str = "ensemble",
    save_results: bool = True
):
    """Predict on multiple images using Ray."""
    global system, RAY_INITIALIZED
    
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not RAY_INITIALIZED:
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if model not in ['clip', 'lgbm', 'ensemble']:
        raise HTTPException(status_code=400, detail="Invalid model specified")
    
    try:
        # Save uploaded files
        saved_files = []
        for image in images:
            if not validate_image_file(image.filename):
                raise HTTPException(status_code=400, detail=f"Invalid image file: {image.filename}")
            
            file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "batch"))
            saved_files.append(file_path)
        
        # Perform batch prediction using Ray
        start_time = time.time()
        
        if model in ['clip', 'lgbm']:
            results = await predict_batch_with_ray(saved_files, model)
        else:  # ensemble
            # For ensemble, we need to get predictions from both models
            clip_results = await predict_batch_with_ray(saved_files, 'clip')
            lgbm_results = await predict_batch_with_ray(saved_files, 'lgbm')
            
            # Combine results
            results = []
            for i in range(len(saved_files)):
                clip_result = clip_results[i]
                lgbm_result = lgbm_results[i]
                
                # Weighted ensemble
                ensemble_prob_cataract = 0.5 * clip_result['probabilities']['cataract'] + 0.5 * lgbm_result['probabilities']['cataract']
                ensemble_prob_normal = 1 - ensemble_prob_cataract
                
                ensemble_class = 1 if ensemble_prob_cataract > 0.5 else 0
                ensemble_confidence = max(ensemble_prob_normal, ensemble_prob_cataract)
                
                results.append({
                    'class': ensemble_class,
                    'class_name': 'cataract' if ensemble_class == 1 else 'normal',
                    'confidence': ensemble_confidence,
                    'probabilities': {
                        'normal': ensemble_prob_normal,
                        'cataract': ensemble_prob_cataract
                    },
                    'ray_worker_id': f"{clip_result.get('ray_worker_id')}+{lgbm_result.get('ray_worker_id')}"
                })
        
        processing_time = time.time() - start_time
        
        # Calculate summary
        total_images = len(results)
        cataract_predictions = sum(1 for r in results if r['class'] == 1)
        normal_predictions = sum(1 for r in results if r['class'] == 0)
        avg_confidence = sum(r['confidence'] for r in results) / total_images
        
        summary = {
            'total_images': total_images,
            'cataract_predictions': cataract_predictions,
            'normal_predictions': normal_predictions,
            'cataract_percentage': (cataract_predictions / total_images) * 100,
            'normal_percentage': (normal_predictions / total_images) * 100,
            'avg_confidence': avg_confidence,
            'throughput': total_images / processing_time
        }
        
        return BatchPredictionResponse(
            total_images=total_images,
            predictions=results,
            summary=summary,
            processing_time=processing_time,
            model_used=model,
            timestamp=datetime.now().isoformat(),
            ray_workers_used=len(RAY_WORKERS) if model == 'ensemble' else 1
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


async def train_clip_background():
    """Background task for CLIP training."""
    global system
    try:
        logger.info("Starting CLIP model training...")
        start_time = time.time()
        
        results = system.train_models(train_clip=True, train_lgbm=False, use_mlflow=True)
        
        training_time = time.time() - start_time
        logger.info(f"CLIP training completed in {training_time:.2f}s")
        
        # Save training status
        status_file = os.path.join(BASE_PATH, "training_status.json")
        with open(status_file, 'w') as f:
            json.dump({
                'clip_training': {
                    'status': 'completed',
                    'training_time': training_time,
                    'metrics': results.get('clip', {}).get('metrics', {}),
                    'timestamp': datetime.now().isoformat()
                }
            }, f)
            
    except Exception as e:
        logger.error(f"CLIP training failed: {str(e)}")
        # Save error status
        status_file = os.path.join(BASE_PATH, "training_status.json")
        with open(status_file, 'w') as f:
            json.dump({
                'clip_training': {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }, f)


async def train_lgbm_background():
    """Background task for LGBM training."""
    global system
    try:
        logger.info("Starting LGBM model training...")
        start_time = time.time()
        
        results = system.train_models(train_clip=False, train_lgbm=True, use_mlflow=True)
        
        training_time = time.time() - start_time
        logger.info(f"LGBM training completed in {training_time:.2f}s")
        
        # Save training status
        status_file = os.path.join(BASE_PATH, "training_status.json")
        with open(status_file, 'w') as f:
            json.dump({
                'lgbm_training': {
                    'status': 'completed',
                    'training_time': training_time,
                    'metrics': results.get('lgbm', {}).get('metrics', {}),
                    'timestamp': datetime.now().isoformat()
                }
            }, f)
            
    except Exception as e:
        logger.error(f"LGBM training failed: {str(e)}")
        # Save error status
        status_file = os.path.join(BASE_PATH, "training_status.json")
        with open(status_file, 'w') as f:
            json.dump({
                'lgbm_training': {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }, f)


@app.post("/train_clip", response_model=TrainingResponse)
async def train_clip(background_tasks: BackgroundTasks):
    """Trigger CLIP model retraining."""
    global system
    
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check if training data exists
    training_data_path = os.path.join(TRAIN_PATH, "processed_images")
    if not os.path.exists(training_data_path):
        raise HTTPException(status_code=400, detail="Training data not found")
    
    # Start background training
    background_tasks.add_task(train_clip_background)
    
    return TrainingResponse(
        status="started",
        model="clip",
        training_time=0.0,
        message="CLIP model retraining triggered in background"
    )


@app.post("/train_lgbm", response_model=TrainingResponse)
async def train_lgbm(background_tasks: BackgroundTasks):
    """Trigger LGBM model retraining."""
    global system
    
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check if training data exists
    training_data_path = os.path.join(TRAIN_PATH, "processed_images")
    if not os.path.exists(training_data_path):
        raise HTTPException(status_code=400, detail="Training data not found")
    
    # Start background training
    background_tasks.add_task(train_lgbm_background)
    
    return TrainingResponse(
        status="started",
        model="lgbm",
        training_time=0.0,
        message="LGBM model retraining triggered in background"
    )


@app.get("/training_status")
async def get_training_status():
    """Get the status of background training tasks."""
    status_file = os.path.join(BASE_PATH, "training_status.json")
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            return json.load(f)
    else:
        return {"status": "no_training_history"}


@app.get("/drift_status", response_model=DriftResponse)
async def drift_status():
    """Check for data drift and model performance."""
    global system
    
    try:
        # Count unlabeled samples
        unlabeled_count = 0
        for root, dirs, files in os.walk(BASE_PATH):
            unlabeled_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))])
        
        # Calculate drift score (simplified - in production, use proper drift detection)
        # This is a mock implementation - replace with actual drift detection logic
        drift_score = min(unlabeled_count / 1000.0, 1.0)  # Mock drift score
        
        # Get model performance metrics (if available)
        model_performance = {}
        if system:
            # You could implement actual performance monitoring here
            model_performance = {
                'clip_loaded': system.clip_inference is not None,
                'lgbm_loaded': system.lgbm_inference is not None,
                'device': system.device
            }
        
        return DriftResponse(
            unlabeled_samples=unlabeled_count,
            drift_score=drift_score,
            last_updated=datetime.now().isoformat(),
            model_performance=model_performance
        )
        
    except Exception as e:
        logger.error(f"Drift status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift status: {str(e)}")


@app.get("/system_info")
async def system_info():
    """Get detailed system information."""
    global system, RAY_INITIALIZED, RAY_WORKERS
    
    info = {
        'system_initialized': system is not None,
        'device': system.device if system else 'unknown',
        'models_loaded': {
            'clip': system.clip_inference is not None if system else False,
            'lgbm': system.lgbm_inference is not None if system else False,
            'batch': system.batch_inference is not None if system else False
        },
        'ray_initialized': RAY_INITIALIZED,
        'ray_workers_available': len(RAY_WORKERS),
        'ray_worker_types': list(RAY_WORKERS.keys()) if RAY_WORKERS else [],
        'directories': {
            'base_path': BASE_PATH,
            'train_path': TRAIN_PATH,
            'models_path': MODELS_PATH
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return info


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "JIVI Cataract Classification API with Ray-based concurrent processing",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "prediction": ["/predict_clip", "/predict_lgbm", "/predict_ensemble", "/predict_batch"],
            "training": ["/train_clip", "/train_lgbm", "/training_status"],
            "monitoring": ["/drift_status", "/system_info"]
        },
        "features": {
            "ray_concurrent_processing": True,
            "async_prediction": True,
            "background_training": True
        }
    }


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
