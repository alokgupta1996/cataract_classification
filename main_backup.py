#!/usr/bin/env python3
"""
Main entry point for JIVI Cataract Classification System.
Provides a unified interface for training, inference, and evaluation.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config import TrainingConfig, create_directories
from src.trainers.clip_trainer import CLIPTrainer
from src.trainers.lgbm_trainer import LGBMTrainer
from src.inference.clip_inference import CLIPInference
from src.inference.lgbm_inference import LGBMInference
from src.inference.batch_inference import BatchInference
from src.utils.evaluation import evaluate_model
from src.utils.logging_config import setup_project_logging, get_logger, log_system_info, log_model_info, log_performance_metrics, log_memory_usage, log_gpu_memory


class JIVICataractSystem:
    """
    Main system class for JIVI Cataract Classification.
    Handles training, inference, and evaluation workflows.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the JIVI Cataract Classification System.
        
        Args:
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        # Setup logging
        self.logger = setup_project_logging()
        log_system_info(self.logger)
        
        self.device = self._get_device(device)
        self.config = TrainingConfig()
        
        # Model instances (loaded on demand)
        self.clip_inference = None
        self.lgbm_inference = None
        self.batch_inference = None
        
        # Training instances
        self.clip_trainer = None
        self.lgbm_trainer = None
        
        self.logger.info(f"🚀 JIVI Cataract Classification System initialized")
        self.logger.info(f"📱 Device: {self.device}")
        log_memory_usage(self.logger)
        log_gpu_memory(self.logger)
        self.logger.info("")
        
    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == 'auto':
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
        
    def load_models(self, load_clip: bool = True, load_lgbm: bool = True):
        """
        Load trained models.
        
        Args:
            load_clip (bool): Whether to load CLIP model
            load_lgbm (bool): Whether to load LGBM model
        """
        self.logger.info("📦 Loading trained models...")
        
        if load_clip:
            try:
                clip_model_path = "models/clip/best_model.pth"
                log_model_info(self.logger, "CLIP", clip_model_path, self.device)
                self.clip_inference = CLIPInference(clip_model_path, device=self.device)
                self.logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load CLIP model: {str(e)}")
                self.clip_inference = None
        
        if load_lgbm:
            try:
                lgbm_model_path = "models/lgbm/trained_model.pkl"
                log_model_info(self.logger, "LGBM", lgbm_model_path, self.device)
                self.lgbm_inference = LGBMInference(lgbm_model_path, device=self.device)
                self.logger.info("✅ LGBM model loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load LGBM model: {str(e)}")
                self.lgbm_inference = None
        
        if load_clip and load_lgbm:
            try:
                self.batch_inference = BatchInference(
                    clip_model_path="models/clip/best_model.pth",
                    lgbm_model_path="models/lgbm/trained_model.pkl",
                    device=self.device
                )
                self.logger.info("✅ Batch inference initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize batch inference: {str(e)}")
                self.batch_inference = None
        
        log_memory_usage(self.logger)
        log_gpu_memory(self.logger)
        
    def train_models(self, model_type: str = 'both', use_mlflow: bool = True):
        """
        Train models.
        
        Args:
            model_type (str): Type of model to train ('clip', 'lgbm', 'both')
            use_mlflow (bool): Whether to use MLflow logging
        """
        self.logger.info(f"🎯 Starting model training for: {model_type}")
        self.logger.info(f"📊 MLflow logging: {'enabled' if use_mlflow else 'disabled'}")
        
        start_time = time.time()
        
        if model_type in ['clip', 'both']:
            self.logger.info("🔄 Training CLIP model...")
            try:
                self.clip_trainer = CLIPTrainer(device=self.device)
                # Use train_with_mlflow method
                config = {
                    "data_path": "processed_images",
                    "n_frozen_layers": 10,
                    "num_epochs": 5,
                    "lr": 1e-4,
                    "batch_size": 32,
                    "val_split": 0.2
                }
                self.clip_trainer.train_with_mlflow(config)
                self.logger.info("✅ CLIP training completed successfully")
            except Exception as e:
                self.logger.error(f"❌ CLIP training failed: {str(e)}")
        
        if model_type in ['lgbm', 'both']:
            self.logger.info("🔄 Training LGBM model...")
            try:
                self.lgbm_trainer = LGBMTrainer(device=self.device)
                self.lgbm_trainer.train_with_mlflow(config)
                self.logger.info("✅ LGBM training completed successfully")
            except Exception as e:
                self.logger.error(f"❌ LGBM training failed: {str(e)}")
        
        training_time = time.time() - start_time
        self.logger.info(f"⏱️ Total training time: {training_time:.2f} seconds")
        log_memory_usage(self.logger)
        log_gpu_memory(self.logger)
        
    def predict_single(self, image_path: str, model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            image_path (str): Path to the image
            model_type (str): Type of model to use ('clip', 'lgbm', 'ensemble')
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        self.logger.info(f"🔮 Making prediction for: {image_path}")
        self.logger.info(f"🤖 Model type: {model_type}")
        
        start_time = time.time()
        
        try:
            if model_type == 'clip' and self.clip_inference:
                result = self.clip_inference.predict_single(image_path)
            elif model_type == 'lgbm' and self.lgbm_inference:
                result = self.lgbm_inference.predict_single(image_path)
            elif model_type == 'ensemble' and self.batch_inference:
                result = self.batch_inference.predict_ensemble_single(image_path)
            else:
                raise ValueError(f"Model type '{model_type}' not available")
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Prediction completed in {processing_time:.3f}s")
            self.logger.info(f"📊 Result: {result['class_name']} (confidence: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Prediction failed after {processing_time:.3f}s: {str(e)}")
            raise
        
    def predict_batch(self, image_paths: List[str], model_type: str = 'ensemble') -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            image_paths (List[str]): List of image paths
            model_type (str): Type of model to use ('clip', 'lgbm', 'ensemble')
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        self.logger.info(f"🔮 Making batch predictions for {len(image_paths)} images")
        self.logger.info(f"🤖 Model type: {model_type}")
        
        start_time = time.time()
        
        try:
            if model_type == 'clip' and self.clip_inference:
                results = self.clip_inference.predict_batch(image_paths)
            elif model_type == 'lgbm' and self.lgbm_inference:
                results = self.lgbm_inference.predict_batch(image_paths)
            elif model_type == 'ensemble' and self.batch_inference:
                results = self.batch_inference.predict_ensemble_parallel(image_paths)
            else:
                raise ValueError(f"Model type '{model_type}' not available")
            
            processing_time = time.time() - start_time
            self.logger.info(f"✅ Batch prediction completed in {processing_time:.3f}s")
            self.logger.info(f"📊 Processed {len(results)} images")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Batch prediction failed after {processing_time:.3f}s: {str(e)}")
            raise
        
    def evaluate_performance(self, test_path: str = "processed_images/test"):
        """
        Evaluate model performance on test data.
        
        Args:
            test_path (str): Path to test data
        """
        self.logger.info(f"📊 Evaluating performance on: {test_path}")
        
        if not os.path.exists(test_path):
            self.logger.error(f"❌ Test path does not exist: {test_path}")
            return
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from performance_test import compare_models_performance
            
            results = compare_models_performance(test_path)
            
            evaluation_time = time.time() - start_time
            self.logger.info(f"✅ Performance evaluation completed in {evaluation_time:.2f}s")
            
            # Log performance metrics
            for model_name, metrics in results.items():
                log_performance_metrics(self.logger, metrics, model_name)
            
            return results
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            self.logger.error(f"❌ Performance evaluation failed after {evaluation_time:.2f}s: {str(e)}")
            raise
        
    def shutdown(self):
        """Cleanup resources."""
        self.logger.info("🔄 Shutting down JIVI Cataract Classification System...")
        
        # Clear model instances
        self.clip_inference = None
        self.lgbm_inference = None
        self.batch_inference = None
        self.clip_trainer = None
        self.lgbm_trainer = None
        
        log_memory_usage(self.logger)
        log_gpu_memory(self.logger)
        self.logger.info("✅ System shutdown completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="JIVI Cataract Classification System")
    parser.add_argument('mode', choices=['train', 'inference', 'evaluate'], 
                       help='Operation mode')
    parser.add_argument('--model', choices=['clip', 'lgbm', 'both'], default='both',
                       help='Model type for training')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--test-path', default='processed_images/test',
                       help='Path to test data for evaluation')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Disable MLflow logging (in train mode)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_project_logging(args.log_level)
    logger.info(f"🎯 Starting JIVI Cataract Classification System in {args.mode} mode")
    
    try:
        # Initialize system
        system = JIVICataractSystem(device=args.device)
        
        if args.mode == 'train':
            system.train_models(model_type=args.model, use_mlflow=not args.no_mlflow)
            
        elif args.mode == 'inference':
            system.load_models(load_clip=True, load_lgbm=True)
            logger.info("✅ Models loaded successfully")
            logger.info("🚀 Ready for inference")
            
        elif args.mode == 'evaluate':
            system.load_models(load_clip=True, load_lgbm=True)
            system.evaluate_performance(args.test_path)
            
        logger.info("✅ Operation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Operation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Operation failed: {str(e)}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
    finally:
        if 'system' in locals():
            system.shutdown()


if __name__ == "__main__":
    main() 