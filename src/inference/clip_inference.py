"""
CLIP Model Inference for cataract classification.
"""

import torch
import torch.nn as nn
import open_clip
import numpy as np
from PIL import Image
import os
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import time
import logging
import pandas as pd

from ..utils.clip_utils import load_clip
from ..utils.image_utils import get_clip_transforms


class CLIPClassifier(nn.Module):
    """
    CLIP-based classifier for cataract detection.
    """
    
    def __init__(self, clip_model, num_classes=2):
        """
        Initialize CLIP classifier.
        
        Args:
            clip_model: Pre-trained CLIP model
            num_classes (int): Number of output classes
        """
        super().__init__()
        self.clip = clip_model.visual  # Only image encoder
        embed_dim = clip_model.text_projection.shape[1]
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Logits
        """
        x = self.clip(x)  # returns image embedding (B, 512)
        return self.classifier(x)


class CLIPInference:
    """
    CLIP Model Inference for cataract classification.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize CLIP inference.
        
        Args:
            model_path (str): Path to the trained CLIP model
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
        """Load the trained CLIP model."""
        try:
            self.logger.info(f"Loading CLIP model from {self.model_path}")
            
            # Load base CLIP model
            self.clip_model, self.preprocess, self.tokenizer, _ = load_clip(self.device)
            
            # Create classifier
            classifier = CLIPClassifier(self.clip_model, num_classes=2)
            
            # Load trained weights with weights_only=False for backward compatibility
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            
            self.model = classifier.to(self.device)
            self.model.eval()
            
            self.logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {str(e)}")
            raise
            
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a path, PIL Image, or numpy array")
            
        # Apply CLIP preprocessing
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return image_tensor
        
    def predict_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict:
        """
        Predict on a single image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Dict: Prediction results with class, probability, and confidence
        """
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            return {
                'class': predicted_class,
                'class_name': 'cataract' if predicted_class == 1 else 'normal',
                'confidence': confidence,
                'probabilities': {
                    'normal': probabilities[0][0].item(),
                    'cataract': probabilities[0][1].item()
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
        # Preprocess all images in batch
        image_tensors = []
        for image in images:
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a path, PIL Image, or numpy array")
                
            image_tensor = self.preprocess(image)
            image_tensors.append(image_tensor)
            
        # Stack tensors
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            results = []
            for i in range(len(images)):
                predicted_class = torch.argmax(probabilities[i]).item()
                confidence = probabilities[i][predicted_class].item()
                
                results.append({
                    'class': predicted_class,
                    'class_name': 'cataract' if predicted_class == 1 else 'normal',
                    'confidence': confidence,
                    'probabilities': {
                        'normal': probabilities[i][0].item(),
                        'cataract': probabilities[i][1].item()
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
        
    def get_embeddings(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Get CLIP embeddings for an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            np.ndarray: CLIP embedding vector
        """
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            # Get embeddings from CLIP visual encoder
            embeddings = self.model.clip(image_tensor)
            return embeddings.cpu().numpy()
            
    def get_embeddings_batch(self, images: List[Union[str, Path, Image.Image, np.ndarray]], 
                           batch_size: int = 32) -> np.ndarray:
        """
        Get CLIP embeddings for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: CLIP embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            image_tensors = []
            for image in batch_images:
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert('RGB')
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image).convert('RGB')
                elif not isinstance(image, Image.Image):
                    raise ValueError("Image must be a path, PIL Image, or numpy array")
                    
                image_tensor = self.preprocess(image)
                image_tensors.append(image_tensor)
                
            # Stack tensors
            batch_tensor = torch.stack(image_tensors).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model.clip(batch_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
        
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