"""
LGBM Model Inference for cataract classification using CLIP embeddings.
"""

import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import time
import logging
import cv2

from ..utils.clip_utils import load_clip


class LGBMInference:
    """
    LGBM Model Inference for cataract classification using CLIP embeddings.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize LGBM inference.
        
        Args:
            model_path (str): Path to the trained LGBM model
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.device = self._get_device(device)
        self.model_path = model_path
        self.model = None
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        self.logger = self._setup_logger()
        
        self._load_model()
        
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
        
    def _load_model(self):
        """Load the trained LGBM model."""
        try:
            self.logger.info(f"Loading LGBM model from {self.model_path}")
            
            # Load LGBM model using joblib (since it was saved with joblib.dump)
            import joblib
            self.model = joblib.load(self.model_path)
                
            # Load CLIP model for feature extraction
            self.clip_model, self.preprocess, self.tokenizer, _ = load_clip(self.device)
            
            self.logger.info("LGBM model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading LGBM model: {str(e)}")
            raise
            
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray], 
                        crop_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            crop_size: Target size for cropping
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path, PIL Image, or numpy array")
            
        # Convert to numpy array
        image_array = np.array(image)
        
        # Crop to pupil region (center crop)
        h, w = image_array.shape[:2]
        crop_h, crop_w = crop_size
        
        # Calculate crop coordinates
        start_h = max(0, (h - crop_h) // 2)
        start_w = max(0, (w - crop_w) // 2)
        end_h = min(h, start_h + crop_h)
        end_w = min(w, start_w + crop_w)
        
        # Crop image
        cropped_image = image_array[start_h:end_h, start_w:end_w]
        
        # Resize to target size
        resized_image = cv2.resize(cropped_image, crop_size)
        
        return resized_image
        
    def extract_clip_features(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract CLIP features from an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            np.ndarray: CLIP embedding vector
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path, PIL Image, or numpy array")
            
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            return features.cpu().numpy().flatten()
            
    def predict_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict:
        """
        Predict on a single image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Dict: Prediction results with class, probability, and confidence
        """
        # Extract CLIP features
        features = self.extract_clip_features(image)
        
        # Reshape for prediction
        features_reshaped = features.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(features_reshaped)[0]
        probabilities = self.model.predict_proba(features_reshaped)[0]
        
        confidence = max(probabilities)
        
        return {
            'class': int(prediction),
            'class_name': 'cataract' if prediction == 1 else 'normal',
            'confidence': float(confidence),
            'probabilities': {
                'normal': float(probabilities[0]),
                'cataract': float(probabilities[1])
            }
        }
        
    def predict_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]], 
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict on a batch of images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            batch_size: Batch size for processing
            
        Returns:
            List[Dict]: List of prediction results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch_images)
            results.extend(batch_results)
            
        return results
        
    def _predict_batch_internal(self, images: List[Union[str, Path, Image.Image, np.ndarray]]) -> List[Dict]:
        """
        Internal method for batch prediction.
        
        Args:
            images: List of images for current batch
            
        Returns:
            List[Dict]: Batch prediction results
        """
        # Extract features for all images in batch
        features_list = []
        for image in images:
            features = self.extract_clip_features(image)
            features_list.append(features)
            
        # Stack features
        features_array = np.vstack(features_list)
        
        # Predict
        predictions = self.model.predict(features_array)
        probabilities = self.model.predict_proba(features_array)
        
        # Format results
        results = []
        for i in range(len(images)):
            prediction = predictions[i]
            probs = probabilities[i]
            confidence = max(probs)
            
            results.append({
                'class': int(prediction),
                'class_name': 'cataract' if prediction == 1 else 'normal',
                'confidence': float(confidence),
                'probabilities': {
                    'normal': float(probs[0]),
                    'cataract': float(probs[1])
                }
            })
            
        return results
        
    def predict_folder(self, folder_path: str, batch_size: int = 32) -> List[Dict]:
        """
        Predict on all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            batch_size: Batch size for processing
            
        Returns:
            List[Dict]: List of prediction results with file paths
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return []
            
        self.logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Predict on all images
        results = self.predict_batch(image_files, batch_size)
        
        # Add file paths to results
        for i, result in enumerate(results):
            result['file_path'] = str(image_files[i])
            
        return results
        
    def get_features(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Get CLIP features for an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            np.ndarray: CLIP feature vector
        """
        return self.extract_clip_features(image)
        
    def get_features_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]], 
                          batch_size: int = 32) -> np.ndarray:
        """
        Get CLIP features for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: CLIP feature vectors
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Extract features for batch
            batch_features = []
            for image in batch_images:
                features = self.extract_clip_features(image)
                batch_features.append(features)
                
            all_features.append(np.vstack(batch_features))
            
        return np.vstack(all_features)
        
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
                import json
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            elif format == 'csv':
                # Flatten results for CSV
                flattened_results = []
                for result in results:
                    flat_result = {
                        'file_path': result.get('file_path', ''),
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
                        'file_path': result.get('file_path', ''),
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