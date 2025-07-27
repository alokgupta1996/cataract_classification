#!/usr/bin/env python3
"""
FastAPI production server with Ray integration for concurrent request processing.
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
from src.utils.logging_config import setup_project_logging, get_logger, log_api_request, log_error_with_context, log_memory_usage, log_gpu_memory

# Configure logging
logger = setup_project_logging()

# Initialize FastAPI app
app = FastAPI(
    title="JIVI Cataract Classification API with Ray",
    description="Production API with Ray-based concurrent processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
system: Optional[JIVICataractSystem] = None
RAY_INITIALIZED = False
RAY_WORKERS = {}

# Paths
BASE_PATH = "customer_data"
os.makedirs(BASE_PATH, exist_ok=True)

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    processing_time: float
    ray_worker_id: Optional[str] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    ray_initialized: bool
    ray_workers_available: int
    timestamp: str

# Ray remote functions
@ray.remote
class PredictionWorker:
    """Ray worker for handling prediction requests."""
    
    def __init__(self, model_path: str, model_type: str):
        self.model_type = model_type
        self.model_path = model_path
        self.inference = None
        self.logger = get_logger(f"ray_worker_{model_type}")
        self._load_model()
        
    def _load_model(self):
        """Load the appropriate model."""
        try:
            self.logger.info(f"🔄 Loading {self.model_type} model in Ray worker...")
            if self.model_type == 'clip':
                from src.inference.clip_inference import CLIPInference
                self.inference = CLIPInference(self.model_path, device='cpu')
            elif self.model_type == 'lgbm':
                from src.inference.lgbm_inference import LGBMInference
                self.inference = LGBMInference(self.model_path, device='cpu')
            self.logger.info(f"✅ {self.model_type} model loaded successfully in Ray worker")
        except Exception as e:
            self.logger.error(f"❌ Failed to load {self.model_type} model in Ray worker: {str(e)}")
            raise
        
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """Predict on a single image."""
        try:
            self.logger.info(f"🔮 Ray worker predicting on: {image_path}")
            start_time = time.time()
            result = self.inference.predict_single(image_path)
            processing_time = time.time() - start_time
            
            result['ray_worker_id'] = ray.get_runtime_context().get_worker_id()
            result['processing_time'] = processing_time
            
            self.logger.info(f"✅ Ray worker prediction completed in {processing_time:.3f}s")
            return result
        except Exception as e:
            self.logger.error(f"❌ Ray worker prediction failed: {str(e)}")
            raise


def initialize_ray():
    """Initialize Ray cluster."""
    global RAY_INITIALIZED, RAY_WORKERS
    
    try:
        logger.info("🚀 Initializing Ray cluster...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                log_to_driver=True
            )
            logger.info("✅ Ray initialized successfully")
        else:
            logger.info("ℹ️ Ray already initialized")
        
        # Create Ray workers
        logger.info("🔄 Creating Ray workers...")
        
        # CLIP worker
        try:
            clip_worker = PredictionWorker.remote("models/clip/best_model.pth", "clip")
            RAY_WORKERS['clip'] = clip_worker
            logger.info("✅ CLIP Ray worker created")
        except Exception as e:
            logger.error(f"❌ Failed to create CLIP Ray worker: {str(e)}")
        
        # LGBM worker
        try:
            lgbm_worker = PredictionWorker.remote("models/lgbm/trained_model.pkl", "lgbm")
            RAY_WORKERS['lgbm'] = lgbm_worker
            logger.info("✅ LGBM Ray worker created")
        except Exception as e:
            logger.error(f"❌ Failed to create LGBM Ray worker: {str(e)}")
        
        RAY_INITIALIZED = len(RAY_WORKERS) > 0
        logger.info(f"📊 Ray workers available: {len(RAY_WORKERS)}")
        
        if RAY_INITIALIZED:
            log_memory_usage(logger)
            log_gpu_memory(logger)
        else:
            logger.error("❌ No Ray workers available")
            
    except Exception as e:
        logger.error(f"❌ Ray initialization failed: {str(e)}")
        RAY_INITIALIZED = False


def initialize_system():
    """Initialize the system."""
    global system
    try:
        logger.info("🔄 Initializing system...")
        system = JIVICataractSystem(device='auto')
        system.load_models(load_clip=True, load_lgbm=True)
        initialize_ray()
        logger.info("✅ System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("🚀 Starting JIVI Cataract Classification API with Ray...")
    success = initialize_system()
    if success:
        logger.info("✅ API startup completed successfully")
    else:
        logger.error("❌ API startup failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🔄 Shutting down API...")
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("✅ Ray shutdown completed")
    except Exception as e:
        logger.error(f"❌ Ray shutdown failed: {str(e)}")
    logger.info("✅ API shutdown completed")


def save_uploaded_file(upload_file: UploadFile, base_dir: str) -> str:
    """Save uploaded file and return path."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(upload_file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(base_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        logger.info(f"💾 File saved: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Failed to save file: {str(e)}")
        raise


def validate_image_file(filename: str) -> bool:
    """Validate image file format."""
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return Path(filename).suffix.lower() in allowed_extensions


async def predict_with_ray(image_path: str, model_type: str) -> Dict[str, Any]:
    """Predict using Ray worker."""
    global RAY_WORKERS
    
    if model_type not in RAY_WORKERS:
        raise ValueError(f"Model type {model_type} not available")
    
    # Submit prediction task to Ray worker
    future = RAY_WORKERS[model_type].predict_single.remote(image_path)
    result = await asyncio.get_event_loop().run_in_executor(
        None, ray.get, future
    )
    
    return result


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    global RAY_INITIALIZED, RAY_WORKERS
    
    logger.info("🏥 Health check requested")
    
    response = HealthResponse(
        status="ok" if RAY_INITIALIZED else "degraded",
        ray_initialized=RAY_INITIALIZED,
        ray_workers_available=len(RAY_WORKERS),
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"📊 Health status: {response.status}, Ray workers: {response.ray_workers_available}")
    return response


@app.post("/predict_clip", response_model=PredictionResponse)
async def predict_clip(image: UploadFile = File(...)):
    """Predict using CLIP model with Ray."""
    global RAY_INITIALIZED
    
    start_time = time.time()
    
    if not RAY_INITIALIZED:
        logger.error("❌ Ray not initialized for CLIP prediction")
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        logger.error(f"❌ Invalid image format: {image.filename}")
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "clip"))
        
        # Perform prediction using Ray
        result = await predict_with_ray(file_path, 'clip')
        processing_time = time.time() - start_time
        
        response = PredictionResponse(
            prediction=result['class_name'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_used='clip',
            processing_time=processing_time,
            ray_worker_id=result.get('ray_worker_id'),
            timestamp=datetime.now().isoformat()
        )
        
        log_api_request(logger, "/predict_clip", "POST", processing_time, 200)
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_error_with_context(logger, e, "CLIP prediction")
        log_api_request(logger, "/predict_clip", "POST", processing_time, 500)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_lgbm", response_model=PredictionResponse)
async def predict_lgbm(image: UploadFile = File(...)):
    """Predict using LGBM model with Ray."""
    global RAY_INITIALIZED
    
    start_time = time.time()
    
    if not RAY_INITIALIZED:
        logger.error("❌ Ray not initialized for LGBM prediction")
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        logger.error(f"❌ Invalid image format: {image.filename}")
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "lgbm"))
        
        # Perform prediction using Ray
        result = await predict_with_ray(file_path, 'lgbm')
        processing_time = time.time() - start_time
        
        response = PredictionResponse(
            prediction=result['class_name'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            model_used='lgbm',
            processing_time=processing_time,
            ray_worker_id=result.get('ray_worker_id'),
            timestamp=datetime.now().isoformat()
        )
        
        log_api_request(logger, "/predict_lgbm", "POST", processing_time, 200)
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_error_with_context(logger, e, "LGBM prediction")
        log_api_request(logger, "/predict_lgbm", "POST", processing_time, 500)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_ensemble", response_model=PredictionResponse)
async def predict_ensemble(image: UploadFile = File(...)):
    """Predict using ensemble model with Ray."""
    global RAY_INITIALIZED
    
    start_time = time.time()
    
    if not RAY_INITIALIZED:
        logger.error("❌ Ray not initialized for ensemble prediction")
        raise HTTPException(status_code=503, detail="Ray not initialized")
    
    if not validate_image_file(image.filename):
        logger.error(f"❌ Invalid image format: {image.filename}")
        raise HTTPException(status_code=400, detail="Invalid image file format")
    
    try:
        # Save uploaded file
        file_path = save_uploaded_file(image, os.path.join(BASE_PATH, "ensemble"))
        
        # Perform ensemble prediction using Ray workers concurrently
        logger.info("🔄 Starting ensemble prediction with Ray workers...")
        
        # Get predictions from both models concurrently
        clip_future = predict_with_ray(file_path, 'clip')
        lgbm_future = predict_with_ray(file_path, 'lgbm')
        
        clip_result, lgbm_result = await asyncio.gather(clip_future, lgbm_future)
        
        # Weighted ensemble (50-50)
        ensemble_prob_cataract = 0.5 * clip_result['probabilities']['cataract'] + 0.5 * lgbm_result['probabilities']['cataract']
        ensemble_prob_normal = 1 - ensemble_prob_cataract
        
        ensemble_class = 1 if ensemble_prob_cataract > 0.5 else 0
        ensemble_confidence = max(ensemble_prob_normal, ensemble_prob_cataract)
        
        processing_time = time.time() - start_time
        
        response = PredictionResponse(
            prediction='cataract' if ensemble_class == 1 else 'normal',
            confidence=ensemble_confidence,
            probabilities={'normal': ensemble_prob_normal, 'cataract': ensemble_prob_cataract},
            model_used='ensemble',
            processing_time=processing_time,
            ray_worker_id=f"{clip_result.get('ray_worker_id')}+{lgbm_result.get('ray_worker_id')}",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Ensemble prediction completed in {processing_time:.3f}s")
        log_api_request(logger, "/predict_ensemble", "POST", processing_time, 200)
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        log_error_with_context(logger, e, "Ensemble prediction")
        log_api_request(logger, "/predict_ensemble", "POST", processing_time, 500)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("🏠 Root endpoint accessed")
    return {
        "message": "JIVI Cataract Classification API with Ray",
        "version": "1.0.0",
        "ray_enabled": True,
        "endpoints": ["/predict_clip", "/predict_lgbm", "/predict_ensemble", "/health"]
    }


if __name__ == "__main__":
    logger.info("🚀 Starting uvicorn server...")
    uvicorn.run("api_ray:app", host="0.0.0.0", port=8001, log_level="info") 