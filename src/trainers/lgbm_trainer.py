"""
LGBM Trainer for cataract classification using CLIP embeddings.
"""

import torch
import pandas as pd
import numpy as np
import time
import random
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import cv2

from ..utils.clip_utils import load_clip, build_dataframe
from ..utils.image_utils import preprocess_folder
from ..utils.evaluation import (
    evaluate_model, log_sample_images, log_roc_curve, 
    log_confusion_matrix, log_metrics_to_mlflow
)


class LGBMTrainer:
    """
    LGBM Trainer for cataract classification using CLIP embeddings.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the LGBM trainer.
        
        Args:
            device (str): Device to run inference on
        """
        self.device = device
        self.model = None
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load_clip_model(self):
        """Load CLIP model for feature extraction."""
        self.clip_model, self.preprocess, self.tokenizer, self.device = load_clip(self.device)
        
    def preprocess_data(self, input_paths, output_paths, size=(512, 512)):
        """
        Preprocess images by cropping to pupil.
        
        Args:
            input_paths (dict): Dictionary with 'normal' and 'cataract' paths
            output_paths (dict): Dictionary with output paths
            size (tuple): Target size for images
        """
        for category in ['normal', 'cataract']:
            if category in input_paths and category in output_paths:
                preprocess_folder(input_paths[category], output_paths[category], size=size)
                
    def build_training_data(self, data_paths, augment=True, n_aug=3, method='crop'):
        """
        Build training dataset with CLIP embeddings.
        
        Args:
            data_paths (dict): Dictionary with 'normal' and 'cataract' paths
            augment (bool): Whether to apply augmentation
            n_aug (int): Number of augmentations per image
            method (str): Augmentation method ('crop', 'compress', or 'hybrid')
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        print("Building training dataset...")
        
        # Build DataFrames for each class
        df_normal = build_dataframe(
            data_paths['normal'], 0, self.preprocess, self.clip_model, 
            device=self.device, augment=augment, n_aug=n_aug, method=method
        )
        df_cataract = build_dataframe(
            data_paths['cataract'], 1, self.preprocess, self.clip_model, 
            device=self.device, augment=augment, n_aug=n_aug, method=method
        )
        
        # Combine and shuffle
        df = pd.concat([df_normal, df_cataract]).astype(float).sample(frac=1).reset_index(drop=True)
        
        # Split features and labels
        X = df.iloc[:, :512]
        y = df['category']
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        
        return X_train, X_val, y_train, y_val
        
    def train_classifier(self, X_train, y_train):
        """
        Train LGBM classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            LGBMClassifier: Trained model
        """
        print("Training LGBM classifier...")
        model = LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        return model
        
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """
        Perform hyperparameter tuning using GridSearchCV with comprehensive parameter grid.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid (dict): Parameter grid for tuning
            
        Returns:
            LGBMClassifier: Best model from grid search
        """
        if param_grid is None:
            param_grid = {
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
            
        print("🔍 Performing comprehensive hyperparameter tuning...")
        print(f"📊 Grid search with {len(param_grid)} parameter combinations")
        
        # Use cross-validation with stratification
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create base model
        base_model = LGBMClassifier(
            n_jobs=1,  # Avoid multiprocessing issues
            verbose=-1,
            random_state=42
        )
        
        # Perform grid search
        grid = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv, 
            scoring='f1_weighted',  # Use F1 score for imbalanced data
            n_jobs=1,  # Single-threaded to avoid OpenMP issues
            verbose=1
        )
        
        print("⏳ Starting grid search (this may take a while)...")
        grid.fit(X_train, y_train)
        
        print(f"✅ Best parameters found: {grid.best_params_}")
        print(f"🎯 Best cross-validation score: {grid.best_score_:.4f}")
        
        return grid.best_estimator_, grid.best_params_
        
    def evaluate_model(self, model, X_val, y_val):
        """
        Evaluate model performance.
        
        Args:
            model: Trained LGBM model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            dict: Evaluation metrics
        """
        return evaluate_model(model, X_val, y_val, model_type='lgbm')
        
    def inference_on_folder(self, folder_path, model, crop=True, strict_preprocess=True, batch_size=16):
        """
        Perform inference on a folder of images.
        
        Args:
            folder_path (str): Path to folder with images
            model: Trained LGBM model
            crop (bool): Whether to crop to pupil
            strict_preprocess (bool): Whether to use strict preprocessing
            batch_size (int): Batch size for inference
            
        Returns:
            pandas.DataFrame: Results with predictions and probabilities
        """
        results = []
        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i in tqdm(range(0, len(image_paths), batch_size), desc="🔍 Batched Inference"):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                if crop:
                    from ..utils.image_utils import crop_to_pupil
                    img_cv = crop_to_pupil(path)
                    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                else:
                    img = Image.open(path).convert("RGB")

                if strict_preprocess:
                    img = self.preprocess(img)
                else:
                    from torchvision import transforms
                    img = transforms.Resize((512, 512))(img)
                    img = transforms.ToTensor()(img)

                images.append(img)

            with torch.no_grad(), torch.cuda.amp.autocast():
                batch = torch.stack(images).to(self.device)
                embeddings = self.clip_model.encode_image(batch)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            preds = model.predict(embeddings)
            probs = model.predict_proba(embeddings)

            for j, path in enumerate(batch_paths):
                results.append({
                    'image_path': path,
                    'prediction': 'cataract' if preds[j] == 1 else 'normal',
                    'prob_cataract': round(probs[j][1], 4),
                    'prob_normal': round(probs[j][0], 4),
                })

        return pd.DataFrame(results)
        
    def train_with_mlflow(self, config):
        """
        Complete training pipeline with MLflow logging (without hyperparameter tuning).
        
        Args:
            config (dict): Configuration dictionary with all parameters
            
        Returns:
            tuple: (best_model, metrics)
        """
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("JIVI_Cataract_Classification")
        
        with mlflow.start_run(run_name=config.get('run_name', 'CLIP_LGBM_Cataract_Classifier')):
            # Log configuration
            mlflow.set_tag("clip_model", "ViT-B-32")
            mlflow.set_tag("classifier", "LGBM")
            mlflow.log_param("device", self.device)
            mlflow.log_param("n_augmentations", config.get('n_augmentations', 3))
            mlflow.log_param("dataset_path", config.get('dataset_path', 'processed_images/train/'))
            
            # Load CLIP model
            self.load_clip_model()
            
            # Step A: Build dataset from original images (skip preprocessing)
            start_train_time = time.time()
            X_train, X_val, y_train, y_val = self.build_training_data(
                config['input_paths'],  # Use original images from processed_images
                augment=config.get('augment', True),
                n_aug=config.get('n_augmentations', 3),
                method=config.get('augmentation_method', 'crop')
            )
            
            # Step B: Train model with default parameters
            print("Training LGBM classifier with default parameters...")
            model = LGBMClassifier(
                random_state=42, 
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            )
            model.fit(X_train, y_train)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "lgbm_model")
            
            # Save model to disk
            import joblib
            model_path = "models/lgbm/trained_model.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, "model_disk")
            
            # Step C: Evaluate model
            metrics = self.evaluate_model(model, X_val, y_val)
            log_metrics_to_mlflow(metrics, prefix="final_")
            log_roc_curve(y_val, metrics['y_probs'], "final_roc_curve.png")
            log_confusion_matrix(y_val, metrics['y_pred'], "final_confusion_matrix.png")
            
            # Step D: Inference on test set
            if config.get('test_inference', True):
                start_infer_time = time.time()
                test_results = self.inference_on_folder(
                    config['test_path'],
                    model,
                    crop=config.get('crop', True),
                    strict_preprocess=config.get('strict_preprocess', True),
                    batch_size=config.get('batch_size', 32)
                )
                end_infer_time = time.time()
                
                mlflow.log_metric("inference_time_sec", end_infer_time - start_infer_time)
                mlflow.log_metric("cataract_detected", (test_results['prob_cataract'] > 0.5).sum())
                
                # Save results
                test_results.to_csv("test_predictions.csv", index=False)
                mlflow.log_artifact("test_predictions.csv")
            
            end_train_time = time.time()
            mlflow.log_metric("total_train_time_sec", end_train_time - start_train_time)
            
            print("\n🎯 MLflow run complete. All results logged.\n")
            
            return model, metrics 