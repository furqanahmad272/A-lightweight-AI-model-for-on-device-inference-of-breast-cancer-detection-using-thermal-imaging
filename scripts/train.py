"""
Model Training Script for Thermal Breast Cancer Detection
Standalone script version of the training notebook
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import argparse
import json
from datetime import datetime

def build_model(input_shape=(96, 96, 3), alpha=0.35):
    """Build ultra-lightweight MobileNetV2 model"""
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
    ])
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation (only applied during training)
    x = data_augmentation(inputs)
    
    # MobileNetV2 base
    base_model = MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=None,
        pooling='avg'
    )
    
    x = base_model(x, training=True)
    
    # Classification head
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

def main(args):
    """Main training function"""
    
    print("="*60)
    print("THERMAL BREAST CANCER DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Configuration
    INPUT_SHAPE = (args.image_size, args.image_size, 3)
    
    print(f"\nConfiguration:")
    print(f"  Image Size: {args.image_size}×{args.image_size}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    model = build_model(input_shape=INPUT_SHAPE, alpha=args.alpha)
    
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated model size: ~{total_params * 4 / 1024:.2f} KB (FP32)")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # Setup callbacks
    os.makedirs(args.model_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(args.model_dir, 'mobilenetv2_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(args.model_dir, 'mobilenetv2_base.h5')
    model.save(model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(args.model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2, default=str)
    
    print(f"✅ Training history saved to {history_path}")
    
    # Final metrics
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best validation AUC: {max(history.history['val_auc']):.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MobileNetV2 for breast cancer detection')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Path to save models')
    parser.add_argument('--image_size', type=int, default=96,
                        help='Image size (default: 96)')
    parser.add_argument('--alpha', type=float, default=0.35,
                        help='MobileNetV2 width multiplier (default: 0.35)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    main(args)
