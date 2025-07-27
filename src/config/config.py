"""
Configuration management for training modules.
"""

import os
import torch


class TrainingConfig:
    """
    Configuration class for training parameters.
    """
    
    def __init__(self):
        # Common parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_augmentations = 3
        self.log_sample_images = True
        self.preprocess_data = True
        
        # Data paths
        self.base_data_path = "processed_images"
        self.processed_data_path = "processed_aug_aug"
        
        # Input paths
        self.input_paths = {
            'normal': f"{self.base_data_path}/train/normal",
            'cataract': f"{self.base_data_path}/train/cataract"
        }
        
        # Output paths
        self.output_paths = {
            'normal': f"{self.processed_data_path}/train/normal",
            'cataract': f"{self.processed_data_path}/train/cataract"
        }
        
        # Test path
        self.test_path = f"{self.base_data_path}/test/cataract"
        
    def get_lgbm_config(self):
        """
        Get configuration for LGBM training.
        
        Returns:
            dict: LGBM training configuration
        """
        return {
            'run_name': 'CLIP_LGBM_Cataract_Classifier',
            'device': self.device,
            'n_augmentations': self.n_augmentations,
            'log_sample_images': self.log_sample_images,
            'preprocess_data': self.preprocess_data,
            'input_paths': self.input_paths,
            'output_paths': self.output_paths,
            'output_path': f"{self.processed_data_path}/train",
            'test_path': self.test_path,
            'crop_size': (512, 512),
            'augment': True,
            'augmentation_method': 'crop',  # 'crop', 'compress', or 'hybrid'
            'test_inference': True,
            'crop': True,
            'strict_preprocess': True,
            'batch_size': 32,
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'num_leaves': [31, 50, 100],
                'min_child_samples': [20, 50, 100],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5],
                'boosting_type': ['gbdt', 'dart'],
                'objective': ['binary'],
                'metric': ['binary_logloss'],
                'is_unbalance': [True, False],
                'random_state': [42],
                'verbose': [-1]
            }
        }
        
    def get_clip_config(self):
        """
        Get configuration for CLIP training.
        
        Returns:
            dict: CLIP training configuration
        """
        return {
            'run_name': 'CLIP_Pytorch_TL_Cataract_Classifier',
            'device': self.device,
            'n_augmentations': self.n_augmentations,
            'log_sample_images': self.log_sample_images,
            'preprocess_data': self.preprocess_data,
            'input_paths': self.input_paths,
            'output_paths': self.output_paths,
            'output_path': f"{self.processed_data_path}/train",
            'crop_size': (224, 224),
            'n_frozen_layers': 10,
            'num_epochs': 5,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'val_split': 0.2
        }


def create_directories():
    """
    Create necessary directories for training.
    """
    dirs = [
        "processed_aug_aug/train/normal",
        "processed_aug_aug/train/cataract",
        "mlruns"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}") 