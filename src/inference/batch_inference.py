"""
Ray-based batch inference for cataract classification models.
"""

import torch
import ray
import time
import logging
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .clip_inference import CLIPInference
from .lgbm_inference import LGBMInference


@ray.remote
class CLIPInferenceWorker:
    """Ray worker for CLIP inference."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.inference = CLIPInference(model_path, device)
        
    def predict_batch(self, images: List[str]) -> List[Dict]:
        """Predict on a batch of images."""
        return self.inference.predict_batch(images)
        
    def get_embeddings_batch(self, images: List[str]) -> np.ndarray:
        """Get embeddings for a batch of images."""
        return self.inference.get_embeddings_batch(images)


@ray.remote
class LGBMInferenceWorker:
    """Ray worker for LGBM inference."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.inference = LGBMInference(model_path, device)
        
    def predict_batch(self, images: List[str]) -> List[Dict]:
        """Predict on a batch of images."""
        return self.inference.predict_batch(images)
        
    def get_features_batch(self, images: List[str]) -> np.ndarray:
        """Get features for a batch of images."""
        return self.inference.get_features_batch(images)


class BatchInference:
    """
    Ray-based batch inference for cataract classification models.
    """
    
    def __init__(self, clip_model_path: str, lgbm_model_path: str, 
                 num_workers: int = 4, device: str = 'auto'):
        """
        Initialize batch inference.
        
        Args:
            clip_model_path: Path to CLIP model
            lgbm_model_path: Path to LGBM model
            num_workers: Number of Ray workers
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.clip_model_path = clip_model_path
        self.lgbm_model_path = lgbm_model_path
        self.num_workers = num_workers
        self.device = self._get_device(device)
        self.logger = self._setup_logger()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=num_workers)
            
        self._setup_workers()
        
    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _setup_workers(self):
        """Setup Ray workers."""
        try:
            self.logger.info(f"Setting up {self.num_workers} Ray workers...")
            
            # Create CLIP workers
            self.clip_workers = [
                CLIPInferenceWorker.remote(self.clip_model_path, self.device)
                for _ in range(self.num_workers)
            ]
            
            # Create LGBM workers
            self.lgbm_workers = [
                LGBMInferenceWorker.remote(self.lgbm_model_path, self.device)
                for _ in range(self.num_workers)
            ]
            
            self.logger.info("Ray workers setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up Ray workers: {str(e)}")
            raise
            
    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get all image files from a folder."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([str(f) for f in Path(folder_path).glob(f'*{ext}')])
            image_files.extend([str(f) for f in Path(folder_path).glob(f'*{ext.upper()}')])
            
        return sorted(image_files)
        
    def _split_batch(self, items: List, num_workers: int) -> List[List]:
        """Split items into batches for workers."""
        batch_size = len(items) // num_workers
        remainder = len(items) % num_workers
        
        batches = []
        start = 0
        
        for i in range(num_workers):
            end = start + batch_size + (1 if i < remainder else 0)
            batches.append(items[start:end])
            start = end
            
        return batches
        
    def predict_clip_parallel(self, folder_path: str, batch_size: int = 32) -> List[Dict]:
        """
        Predict using CLIP model with parallel processing.
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Batch size per worker
            
        Returns:
            List[Dict]: Prediction results
        """
        image_files = self._get_image_files(folder_path)
        
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []
            
        self.logger.info(f"Processing {len(image_files)} images with CLIP using {self.num_workers} workers")
        
        # Split images among workers
        worker_batches = self._split_batch(image_files, self.num_workers)
        
        # Submit tasks to workers
        futures = []
        for i, batch in enumerate(worker_batches):
            if batch:  # Only submit if batch is not empty
                future = self.clip_workers[i].predict_batch.remote(batch)
                futures.append(future)
                
        # Collect results
        results = []
        for future in ray.get(futures):
            results.extend(future)
            
        # Add file paths to results
        for i, result in enumerate(results):
            result['file_path'] = image_files[i]
            
        return results
        
    def predict_lgbm_parallel(self, folder_path: str, batch_size: int = 32) -> List[Dict]:
        """
        Predict using LGBM model with parallel processing.
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Batch size per worker
            
        Returns:
            List[Dict]: Prediction results
        """
        image_files = self._get_image_files(folder_path)
        
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []
            
        self.logger.info(f"Processing {len(image_files)} images with LGBM using {self.num_workers} workers")
        
        # Split images among workers
        worker_batches = self._split_batch(image_files, self.num_workers)
        
        # Submit tasks to workers
        futures = []
        for i, batch in enumerate(worker_batches):
            if batch:  # Only submit if batch is not empty
                future = self.lgbm_workers[i].predict_batch.remote(batch)
                futures.append(future)
                
        # Collect results
        results = []
        for future in ray.get(futures):
            results.extend(future)
            
        # Add file paths to results
        for i, result in enumerate(results):
            result['file_path'] = image_files[i]
            
        return results
        
    def predict_ensemble_parallel(self, folder_path: str, batch_size: int = 32, 
                                weights: Tuple[float, float] = (0.5, 0.5)) -> List[Dict]:
        """
        Predict using both models with ensemble voting.
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Batch size per worker
            weights: Weights for CLIP and LGBM predictions (clip_weight, lgbm_weight)
            
        Returns:
            List[Dict]: Ensemble prediction results
        """
        image_files = self._get_image_files(folder_path)
        
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []
            
        self.logger.info(f"Processing {len(image_files)} images with ensemble using {self.num_workers} workers")
        
        # Get predictions from both models
        clip_results = self.predict_clip_parallel(folder_path, batch_size)
        lgbm_results = self.predict_lgbm_parallel(folder_path, batch_size)
        
        # Create file to result mapping
        clip_dict = {result['file_path']: result for result in clip_results}
        lgbm_dict = {result['file_path']: result for result in lgbm_results}
        
        # Combine predictions
        ensemble_results = []
        for file_path in image_files:
            clip_pred = clip_dict.get(file_path)
            lgbm_pred = lgbm_dict.get(file_path)
            
            if clip_pred and lgbm_pred:
                # Weighted ensemble
                clip_weight, lgbm_weight = weights
                
                # Weighted probability
                ensemble_prob_normal = (
                    clip_weight * clip_pred['probabilities']['normal'] +
                    lgbm_weight * lgbm_pred['probabilities']['normal']
                )
                ensemble_prob_cataract = (
                    clip_weight * clip_pred['probabilities']['cataract'] +
                    lgbm_weight * lgbm_pred['probabilities']['cataract']
                )
                
                # Determine class
                ensemble_class = 1 if ensemble_prob_cataract > ensemble_prob_normal else 0
                ensemble_confidence = max(ensemble_prob_normal, ensemble_prob_cataract)
                
                ensemble_results.append({
                    'file_path': file_path,
                    'class': ensemble_class,
                    'class_name': 'cataract' if ensemble_class == 1 else 'normal',
                    'confidence': ensemble_confidence,
                    'probabilities': {
                        'normal': ensemble_prob_normal,
                        'cataract': ensemble_prob_cataract
                    },
                    'clip_prediction': clip_pred,
                    'lgbm_prediction': lgbm_pred
                })
                
        return ensemble_results
        
    def save_results(self, results: List[Dict], output_path: str, format: str = 'json'):
        """
        Save prediction results to file.
        
        Args:
            results: List of prediction results
            output_path: Path to save results
            format: Output format ('json', 'csv', 'excel')
        """
        try:
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            elif format == 'csv':
                # Flatten results for CSV
                flattened_results = []
                for result in results:
                    flat_result = {
                        'file_path': result['file_path'],
                        'class': result['class'],
                        'class_name': result['class_name'],
                        'confidence': result['confidence'],
                        'prob_normal': result['probabilities']['normal'],
                        'prob_cataract': result['probabilities']['cataract']
                    }
                    flattened_results.append(flat_result)
                    
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
                
            elif format == 'excel':
                # Flatten results for Excel
                flattened_results = []
                for result in results:
                    flat_result = {
                        'file_path': result['file_path'],
                        'class': result['class'],
                        'class_name': result['class_name'],
                        'confidence': result['confidence'],
                        'prob_normal': result['probabilities']['normal'],
                        'prob_cataract': result['probabilities']['cataract']
                    }
                    flattened_results.append(flat_result)
                    
                df = pd.DataFrame(flattened_results)
                df.to_excel(output_path, index=False)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
            
    def get_performance_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate performance metrics from results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Dict: Performance metrics
        """
        if not results:
            return {}
            
        total_images = len(results)
        cataract_predictions = sum(1 for r in results if r['class'] == 1)
        normal_predictions = sum(1 for r in results if r['class'] == 0)
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_cataract_prob = np.mean([r['probabilities']['cataract'] for r in results])
        avg_normal_prob = np.mean([r['probabilities']['normal'] for r in results])
        
        return {
            'total_images': total_images,
            'cataract_predictions': cataract_predictions,
            'normal_predictions': normal_predictions,
            'cataract_percentage': (cataract_predictions / total_images) * 100,
            'normal_percentage': (normal_predictions / total_images) * 100,
            'avg_confidence': avg_confidence,
            'avg_cataract_probability': avg_cataract_prob,
            'avg_normal_probability': avg_normal_prob
        }
        
    def shutdown(self):
        """Shutdown Ray workers."""
        try:
            ray.shutdown()
            self.logger.info("Ray workers shutdown completed")
        except Exception as e:
            self.logger.error(f"Error shutting down Ray workers: {str(e)}")