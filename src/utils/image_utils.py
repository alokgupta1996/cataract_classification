"""
Image processing utilities for cataract classification.
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import glob
from torchvision import transforms


def crop_to_pupil(image_path, target_size=(512, 512)):
    """
    Crop image to pupil region using OpenCV.
    
    Args:
        image_path (str): Path to image
        target_size (tuple): Target size for output image
        
    Returns:
        numpy.ndarray: Cropped image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None
    
    # Convert to grayscale for pupil detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Hough Circle Transform to detect pupil
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=100
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Get the largest circle (most likely the pupil)
        largest_circle = max(circles, key=lambda x: x[2])
        x, y, radius = largest_circle
        
        # Calculate crop boundaries
        crop_size = min(radius * 3, min(img.shape[:2]) // 2)
        x1 = max(0, x - crop_size)
        y1 = max(0, y - crop_size)
        x2 = min(img.shape[1], x + crop_size)
        y2 = min(img.shape[0], y + crop_size)
        
        # Crop image
        cropped = img[y1:y2, x1:x2]
        
        # Resize to target size
        resized = cv2.resize(cropped, target_size)
        
        return resized
    else:
        print(f"⚠️ Pupil not detected in {image_path}, using full image.")
        return cv2.resize(img, target_size)


def compress_to_size(image_path, target_size=(224, 224)):
    """
    Compress image to target size without cropping (maintains aspect ratio).
    
    Args:
        image_path (str): Path to image
        target_size (tuple): Target size for output image
        
    Returns:
        numpy.ndarray: Compressed image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None
    
    # Calculate aspect ratios
    h, w = img.shape[:2]
    target_h, target_w = target_size
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h
    
    # Determine resize dimensions to maintain aspect ratio
    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target, fit to width
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        # Image is taller than target, fit to height
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create target-sized canvas with black padding
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image on canvas
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return canvas


def get_clip_transforms():
    """
    Get CLIP-specific transforms.
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])


def albumentations_aug(image, n_augmentations=3):
    """
    Apply Albumentations augmentations to image.
    
    Args:
        image (numpy.ndarray): Input image
        n_augmentations (int): Number of augmentations to apply
        
    Returns:
        list: List of augmented images
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(p=0.2),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(p=0.3),
            A.Emboss(p=0.3),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.OneOf([
            A.Perspective(scale=(0.05, 0.1)),
            A.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1)),
        ], p=0.2),
    ])
    
    augmented_images = []
    for _ in range(n_augmentations):
        augmented = transform(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images


def preprocess_folder(input_folder, output_folder, size=(512, 512), method='crop'):
    """
    Preprocess all images in a folder.
    
    Args:
        input_folder (str): Input folder path
        output_folder (str): Output folder path
        size (tuple): Target size for images
        method (str): Processing method ('crop' or 'compress')
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    print(f"📁 Processing {len(image_files)} images from {input_folder}")
    
    for image_path in tqdm(image_files, desc=f"Processing {os.path.basename(input_folder)}"):
        try:
            # Choose processing method
            if method == 'compress':
                processed_img = compress_to_size(image_path, size)
            else:  # default to crop
                processed_img = crop_to_pupil(image_path, size)
            
            if processed_img is not None:
                # Save processed image
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed_img)
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")


def create_compressed_augmentations(image_path, target_size=(224, 224), n_augmentations=3):
    """
    Create compressed augmentations for an image.
    
    Args:
        image_path (str): Path to image
        target_size (tuple): Target size for output
        n_augmentations (int): Number of augmentations
        
    Returns:
        list: List of compressed and augmented images
    """
    # First compress the image
    compressed = compress_to_size(image_path, target_size)
    if compressed is None:
        return []
    
    # Then apply augmentations
    augmented = albumentations_aug(compressed, n_augmentations)
    
    return [compressed] + augmented  # Include original compressed image


def create_cropped_augmentations(image_path, target_size=(512, 512), n_augmentations=3):
    """
    Create cropped augmentations for an image.
    
    Args:
        image_path (str): Path to image
        target_size (tuple): Target size for output
        n_augmentations (int): Number of augmentations
        
    Returns:
        list: List of cropped and augmented images
    """
    # First crop the image
    cropped = crop_to_pupil(image_path, target_size)
    if cropped is None:
        return []
    
    # Then apply augmentations
    augmented = albumentations_aug(cropped, n_augmentations)
    
    return [cropped] + augmented  # Include original cropped image 