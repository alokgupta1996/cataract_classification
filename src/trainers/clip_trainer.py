"""
CLIP Trainer for fine-tuning CLIP models for cataract classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import time
import random
import mlflow
import mlflow.pytorch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from glob import glob

from ..utils.clip_utils import load_clip
from ..utils.image_utils import preprocess_folder, albumentations_aug, get_clip_transforms
from ..utils.dataset import CataractDataset
from ..utils.evaluation import (
    evaluate_model, log_sample_images, log_roc_curve, 
    log_confusion_matrix, log_metrics_to_mlflow
)


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


def freeze_clip_layers(clip_visual, n_frozen_layers=10):
    """
    Freezes first n_frozen_layers of the CLIP ViT encoder.
    
    Args:
        clip_visual: CLIP visual encoder
        n_frozen_layers (int): Number of layers to freeze
    """
    for name, param in clip_visual.named_parameters():
        if 'transformer.resblocks' in name:
            block_idx = int(name.split('.')[2])
            if block_idx < n_frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif 'ln_post' in name or 'proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False  # patch embedding, pos_embed, etc.


class CLIPTrainer:
    """
    CLIP Trainer for fine-tuning CLIP models for cataract classification.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the CLIP trainer.
        
        Args:
            device (str): Device to run training on
        """
        self.device = device
        self.model = None
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load_clip_model(self):
        """Load CLIP model."""
        self.clip_model, self.preprocess, self.tokenizer, self.device = load_clip(self.device)
        
    def create_model(self, n_frozen_layers=10):
        """
        Create CLIP classifier model.
        
        Args:
            n_frozen_layers (int): Number of CLIP layers to freeze
            
        Returns:
            CLIPClassifier: Initialized model
        """
        model = CLIPClassifier(self.clip_model).to(self.device)
        freeze_clip_layers(model.clip, n_frozen_layers=n_frozen_layers)
        return model
        
    def preprocess_data(self, input_paths, output_paths, size=(224, 224)):
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
                
    def create_data_loaders(self, data_path, batch_size=32, val_split=0.2):
        """
        Create training and validation data loaders.
        
        Args:
            data_path (str): Path to processed data
            batch_size (int): Batch size
            val_split (float): Validation split ratio
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Create dataset
        dataset = CataractDataset(
            root_dir=data_path,
            transform=get_clip_transforms(),
            augment_fn=albumentations_aug,
            target_size=(224, 224)
        )
        
        # Split dataset
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
        
    def train_one_epoch(self, model, dataloader, optimizer, criterion):
        """
        Train for one epoch.
        
        Args:
            model: CLIP classifier model
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            tuple: (avg_loss, accuracy)
        """
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, model, dataloader, criterion):
        """
        Validate model.
        
        Args:
            model: CLIP classifier model
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            tuple: (avg_loss, accuracy)
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
        
    def train_model(self, train_loader, val_loader, n_frozen_layers=10, num_epochs=5, lr=1e-4):
        """
        Train the CLIP classifier.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_frozen_layers (int): Number of CLIP layers to freeze
            num_epochs (int): Number of training epochs
            lr (float): Learning rate
            
        Returns:
            CLIPClassifier: Trained model
        """
        # Create model
        model = self.create_model(n_frozen_layers=n_frozen_layers)
        
        # Define optimizer and loss
        criterion = nn.CrossEntropyLoss()
        
        head_params = model.classifier.parameters()
        clip_params = [p for p in model.clip.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW([
            {'params': clip_params, 'lr': lr / 10},
            {'params': head_params, 'lr': lr}
        ])
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs} | Frozen CLIP layers: {n_frozen_layers}")
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
            
        return model
        
    def evaluate_model(self, model, val_loader):
        """
        Evaluate model performance.
        
        Args:
            model: Trained CLIP model
            val_loader: Validation data loader
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics manually
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'y_pred': all_preds,
            'y_probs': all_probs,
            'y_true': all_labels
        }
        
    def train_with_mlflow(self, config):
        """
        Complete training pipeline with MLflow logging.
        
        Args:
            config (dict): Configuration dictionary with all parameters
            
        Returns:
            tuple: (trained_model, metrics)
        """
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("JIVI_Cataract_Classification")
        
        with mlflow.start_run(run_name=config.get('run_name', 'CLIP_Pytorch_TL_Cataract_Classifier')):
            # Log configuration
            mlflow.set_tag("clip_model", "ViT-B-32")
            mlflow.set_tag("classifier", "CLIP_FineTuned")
            mlflow.log_param("device", self.device)
            mlflow.log_param("n_augmentations", config.get('n_augmentations', 3))
            mlflow.log_param("n_frozen_layers", config.get('n_frozen_layers', 10))
            mlflow.log_param("num_epochs", config.get('num_epochs', 5))
            mlflow.log_param("learning_rate", config.get('learning_rate', 1e-4))
            
            # Load CLIP model
            self.load_clip_model()
            
            # Step A: Create data loaders using original images
            print("Creating data loaders from original images...")
            train_loader, val_loader = self.create_data_loaders(
                config['input_paths']['normal'].replace('/normal', ''),  # Use processed_images/train
                batch_size=config.get('batch_size', 32),
                val_split=config.get('val_split', 0.2)
            )
            
            # Step B: Train model
            start_train_time = time.time()
            trained_model = self.train_model(
                train_loader, val_loader,
                n_frozen_layers=config.get('n_frozen_layers', 10),
                num_epochs=config.get('num_epochs', 5),
                lr=config.get('learning_rate', 1e-4)
            )
            
            # Step C: Evaluate model
            metrics = self.evaluate_model(trained_model, val_loader)
            log_metrics_to_mlflow(metrics, prefix="final_")
            log_roc_curve(metrics['y_true'], metrics['y_probs'], "final_roc_curve.png")
            log_confusion_matrix(metrics['y_true'], metrics['y_pred'], "final_confusion_matrix.png")
            
            # Log model to MLflow
            mlflow.pytorch.log_model(trained_model, "clip_model")
            
            # Save model to disk
            import torch
            model_path = "models/clip/best_model.pth"
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'config': config,
                'metrics': metrics
            }, model_path)
            mlflow.log_artifact(model_path, "model_disk")
            
            end_train_time = time.time()
            mlflow.log_metric("total_train_time_sec", end_train_time - start_train_time)
            
            print("\n🎯 MLflow run complete. All results logged.\n")
            
            return trained_model, metrics 