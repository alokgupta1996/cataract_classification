"""
CLIP utilities for loading models and building datasets.
"""

import torch
import open_clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import glob
import cv2
from ..utils.image_utils import (
    create_cropped_augmentations, 
    create_compressed_augmentations,
    albumentations_aug
)


def load_clip(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load CLIP model and preprocessing.
    
    Args:
        device (str): Device to load model on
        
    Returns:
        tuple: (model, preprocess, tokenizer, device)
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    model = model.to(device)
    return model, preprocess, tokenizer, device


def build_dataframe(data_path, category, preprocess, clip_model, device, augment=True, n_aug=3, method='crop'):
    """
    Build DataFrame with CLIP embeddings from images.
    
    Args:
        data_path (str): Path to image directory
        category (int): Category label (0 for normal, 1 for cataract)
        preprocess: CLIP preprocessing function
        clip_model: CLIP model
        device (str): Device to run inference on
        augment (bool): Whether to apply augmentation
        n_aug (int): Number of augmentations per image
        method (str): Augmentation method ('crop' or 'compress')
        
    Returns:
        pandas.DataFrame: DataFrame with embeddings and labels
    """
    print(f"📊 Building DataFrame for category {category} from {data_path}")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_path, ext)))
        image_files.extend(glob.glob(os.path.join(data_path, ext.upper())))
    
    if not image_files:
        print(f"⚠️ No images found in {data_path}")
        return pd.DataFrame()
    
    embeddings_list = []
    categories_list = []
    
    for image_path in tqdm(image_files, desc=f"Processing category {category}"):
        try:
            if augment:
                # Choose augmentation method
                if method == 'compress':
                    # Use compression method (maintains aspect ratio)
                    augmented_images = create_compressed_augmentations(
                        image_path, target_size=(224, 224), n_augmentations=n_aug
                    )
                else:
                    # Use cropping method (crops to pupil)
                    augmented_images = create_cropped_augmentations(
                        image_path, target_size=(512, 512), n_augmentations=n_aug
                    )
                
                # Process each augmented image
                for i, img_array in enumerate(augmented_images):
                    # Convert numpy array to PIL Image
                    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                    
                    # Apply CLIP preprocessing
                    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = clip_model.encode_image(img_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embedding = embedding.cpu().numpy().flatten()
                    
                    embeddings_list.append(embedding)
                    categories_list.append(category)
                    
            else:
                # No augmentation - process single image
                img_pil = Image.open(image_path).convert("RGB")
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = clip_model.encode_image(img_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    embedding = embedding.cpu().numpy().flatten()
                
                embeddings_list.append(embedding)
                categories_list.append(category)
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    # Create DataFrame
    if embeddings_list:
        df = pd.DataFrame(embeddings_list)
        df['category'] = categories_list
        return df
    else:
        return pd.DataFrame()


def build_dataframe_advanced(data_path, category, preprocess, clip_model, device, 
                           augment=True, n_aug=3, method='crop', target_size=(224, 224)):
    """
    Advanced DataFrame builder with flexible augmentation methods.
    
    Args:
        data_path (str): Path to image directory
        category (int): Category label (0 for normal, 1 for cataract)
        preprocess: CLIP preprocessing function
        clip_model: CLIP model
        device (str): Device to run inference on
        augment (bool): Whether to apply augmentation
        n_aug (int): Number of augmentations per image
        method (str): Augmentation method ('crop', 'compress', or 'hybrid')
        target_size (tuple): Target size for processing
        
    Returns:
        pandas.DataFrame: DataFrame with embeddings and labels
    """
    print(f"📊 Building advanced DataFrame for category {category} using {method} method")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_path, ext)))
        image_files.extend(glob.glob(os.path.join(data_path, ext.upper())))
    
    if not image_files:
        print(f"⚠️ No images found in {data_path}")
        return pd.DataFrame()
    
    embeddings_list = []
    categories_list = []
    
    for image_path in tqdm(image_files, desc=f"Processing category {category}"):
        try:
            if augment:
                if method == 'hybrid':
                    # Use both cropping and compression methods
                    cropped_images = create_cropped_augmentations(
                        image_path, target_size=(512, 512), n_augmentations=n_aug//2
                    )
                    compressed_images = create_compressed_augmentations(
                        image_path, target_size=target_size, n_augmentations=n_aug//2
                    )
                    augmented_images = cropped_images + compressed_images
                elif method == 'compress':
                    augmented_images = create_compressed_augmentations(
                        image_path, target_size=target_size, n_augmentations=n_aug
                    )
                else:  # default to crop
                    augmented_images = create_cropped_augmentations(
                        image_path, target_size=(512, 512), n_augmentations=n_aug
                    )
                
                # Process each augmented image
                for img_array in augmented_images:
                    # Convert numpy array to PIL Image
                    img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                    
                    # Apply CLIP preprocessing
                    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = clip_model.encode_image(img_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embedding = embedding.cpu().numpy().flatten()
                    
                    embeddings_list.append(embedding)
                    categories_list.append(category)
                    
            else:
                # No augmentation - process single image
                img_pil = Image.open(image_path).convert("RGB")
                img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = clip_model.encode_image(img_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    embedding = embedding.cpu().numpy().flatten()
                
                embeddings_list.append(embedding)
                categories_list.append(category)
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    # Create DataFrame
    if embeddings_list:
        df = pd.DataFrame(embeddings_list)
        df['category'] = categories_list
        return df
    else:
        return pd.DataFrame()


def get_clip_embeddings(images, model, preprocess, device='cuda'):
    """
    Get CLIP embeddings for a batch of images.
    
    Args:
        images (list): List of PIL images
        model: CLIP model
        preprocess: CLIP preprocessing function
        device (str): Device to run inference on
    
    Returns:
        numpy.ndarray: Normalized embeddings
    """
    with torch.no_grad(), torch.cuda.amp.autocast():
        tensors = torch.stack([preprocess(img) for img in images]).to(device)
        embeddings = model.encode_image(tensors)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy() 