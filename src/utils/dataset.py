"""
Dataset classes for both LGBM and CLIP trainers.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from glob import glob
from .image_utils import crop_to_pupil


class CataractDataset(Dataset):
    """
    Dataset class for cataract classification.
    """
    
    def __init__(self, root_dir, transform=None, augment_fn=None, target_size=(224, 224)):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Path to 'processed_images/train'
            transform: torchvision transforms to convert to tensor and normalize
            augment_fn: albumentations pipeline (optional)
            target_size (tuple): Target size for images
        """
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'normal': 0, 'cataract': 1}

        for label_name in ['normal', 'cataract']:
            folder = os.path.join(root_dir, label_name)
            paths = glob(f"{folder}/*.png") + glob(f"{folder}/*.jpg") + glob(f"{folder}/*.jpeg")
            self.image_paths.extend(paths)
            self.labels.extend([self.class_to_idx[label_name]] * len(paths))

        self.transform = transform
        self.augment_fn = augment_fn
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Crop to pupil
        image = crop_to_pupil(image_path, target_size=self.target_size)  # np.array in BGR
        
        # Handle case where crop_to_pupil returns None
        if image is None:
            # Load original image and resize
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.target_size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations (optional)
        if self.augment_fn:
            # albumentations_aug returns a list, so we take the first one
            augmented_images = self.augment_fn(image, n_augmentations=1)
            image = augmented_images[0]

        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


class CLIPEmbeddingDataset(Dataset):
    """
    Dataset class for CLIP embeddings (used with LGBM).
    """
    
    def __init__(self, embeddings, labels):
        """
        Initialize the dataset with pre-computed embeddings.
        
        Args:
            embeddings (numpy.ndarray): CLIP embeddings
            labels (numpy.ndarray): Corresponding labels
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx] 