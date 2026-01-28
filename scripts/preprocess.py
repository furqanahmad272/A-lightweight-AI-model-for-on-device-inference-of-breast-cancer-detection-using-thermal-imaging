"""
Data Preprocessing Script for Thermal Breast Cancer Detection
Standalone script version of the preprocessing notebook
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse


def preprocess_image(img, target_size=96):
    """Preprocess a single image"""
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    return img_normalized


def load_images_from_folder(folder):
    """Load all images from a folder"""
    images = []
    filenames = []
    
    for filename in tqdm(os.listdir(folder), desc=f"Loading {os.path.basename(folder)}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    
    return images, filenames


def main(args):
    """Main preprocessing function"""
    
    print("="*60)
    print("THERMAL BREAST CANCER DETECTION - DATA PREPROCESSING")
    print("="*60)
    
    # Configuration
    IMAGE_SIZE = args.image_size
    RAW_DATA_DIR = args.data_dir
    PROCESSED_DATA_DIR = args.output_dir
    SEED = args.seed
    
    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print(f"\nConfiguration:")
    print(f"  Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  Raw Data Directory: {RAW_DATA_DIR}")
    print(f"  Output Directory: {PROCESSED_DATA_DIR}")
    print(f"  Random Seed: {SEED}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    healthy_path = os.path.join(RAW_DATA_DIR, 'healthy')
    cancer_path = os.path.join(RAW_DATA_DIR, 'cancer')
    
    if not os.path.exists(healthy_path) or not os.path.exists(cancer_path):
        print(f"Error: Required folders not found!")
        print(f"Please ensure the following structure exists:")
        print(f"  {RAW_DATA_DIR}/healthy/")
        print(f"  {RAW_DATA_DIR}/cancer/")
        return
    
    healthy_images, healthy_files = load_images_from_folder(healthy_path)
    cancer_images, cancer_files = load_images_from_folder(cancer_path)
    
    print(f"\nHealthy images: {len(healthy_images)}")
    print(f"Cancer images: {len(cancer_images)}")
    print(f"Total images: {len(healthy_images) + len(cancer_images)}")
    
    if len(healthy_images) == 0 or len(cancer_images) == 0:
        print("Error: No images found in one or both categories!")
        return
    
    # Preprocess images
    print("\n" + "="*60)
    print("PREPROCESSING IMAGES")
    print("="*60)
    
    print("Processing healthy images...")
    healthy_processed = [preprocess_image(img, IMAGE_SIZE) for img in tqdm(healthy_images)]
    
    print("Processing cancer images...")
    cancer_processed = [preprocess_image(img, IMAGE_SIZE) for img in tqdm(cancer_images)]
    
    # Create labels and combine
    X_healthy = np.array(healthy_processed)
    X_cancer = np.array(cancer_processed)
    
    y_healthy = np.zeros(len(X_healthy))
    y_cancer = np.ones(len(X_cancer))
    
    X = np.concatenate([X_healthy, X_cancer], axis=0)
    y = np.concatenate([y_healthy, y_cancer], axis=0)
    
    # Shuffle
    X, y = shuffle(X, y, random_state=SEED)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED, stratify=y
    )
    
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=SEED, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print(f"\nClass distribution:")
    print(f"  Train - Healthy: {np.sum(y_train == 0)}, Cancer: {np.sum(y_train == 1)}")
    print(f"  Val   - Healthy: {np.sum(y_val == 0)}, Cancer: {np.sum(y_val == 1)}")
    print(f"  Test  - Healthy: {np.sum(y_test == 0)}, Cancer: {np.sum(y_test == 1)}")
    
    # Save preprocessed data
    print("\n" + "="*60)
    print("SAVING PREPROCESSED DATA")
    print("="*60)
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    print(f"✅ Preprocessed data saved to {PROCESSED_DATA_DIR}")
    print("="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess thermal imaging data')
    parser.add_argument('--data_dir', type=str, default='../data/raw/dmr_ir',
                        help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                        help='Path to output directory')
    parser.add_argument('--image_size', type=int, default=96,
                        help='Target image size (default: 96)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    main(args)