"""
Evaluation utilities for model assessment and MLflow logging.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import mlflow
from PIL import Image
import os
from .image_utils import crop_to_pupil


def evaluate_model(model, X_val, y_val, model_type='lgbm'):
    """
    Evaluate model performance with various metrics.
    
    Args:
        model: Trained model (LGBM or CLIP)
        X_val: Validation features
        y_val: Validation labels
        model_type (str): Type of model ('lgbm' or 'clip')
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    if model_type == 'lgbm':
        y_pred = model.predict(X_val)
        y_probs = model.predict_proba(X_val)[:, 1]
    else:
        # For CLIP model, assuming it returns logits
        import torch
        with torch.no_grad():
            outputs = model(X_val)
            y_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred = (y_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'classification_report': report,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'y_true': y_val
    }


def log_sample_images(image_path, n_augmentations=3):
    """
    Log sample images and their augmentations to MLflow.
    
    Args:
        image_path (str): Path to the image
        n_augmentations (int): Number of augmentations to create
    """
    try:
        # Load original image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # Create augmentations
        from .image_utils import albumentations_aug
        aug_images = albumentations_aug(image_array, n_augmentations=n_augmentations)
        
        # Log original image
        mlflow.log_image(image_array, f"sample_original_{os.path.basename(image_path)}")
        
        # Log augmented images
        for i, aug_img in enumerate(aug_images):
            mlflow.log_image(aug_img, f"sample_aug_{i}_{os.path.basename(image_path)}")
            
    except Exception as e:
        print(f"⚠️ Error logging sample images: {e}")


def log_roc_curve(y_true, y_probs, file_path="roc_curve.png"):
    """
    Generate and log ROC curve to MLflow.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        file_path (str): Path to save ROC curve image
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(file_path)
    mlflow.log_artifact(file_path)
    mlflow.log_metric("val_roc_auc", roc_auc)


def log_confusion_matrix(y_true, y_pred, file_path="confusion_matrix.png"):
    """
    Generate and log confusion matrix to MLflow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        file_path (str): Path to save confusion matrix image
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(file_path)
    mlflow.log_artifact(file_path)


def log_metrics_to_mlflow(metrics, prefix=""):
    """
    Log evaluation metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        prefix (str): Prefix for metric names
    """
    metric_names = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    
    for metric_name in metric_names:
        if metric_name in metrics:
            mlflow.log_metric(f"{prefix}{metric_name}", metrics[metric_name])
    
    # Log detailed classification report metrics
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                for metric_name, value in class_metrics.items():
                    if metric_name in ['precision', 'recall', 'f1-score']:
                        mlflow.log_metric(f"{prefix}{class_name}_{metric_name}", value) 