"""
Model Quantization Script for MAX78000 Deployment
Converts trained model to INT8 quantized TFLite format
"""

import os
import numpy as np
import tensorflow as tf
import argparse

def representative_dataset_gen(data_path, num_samples=100):
    """Generator for representative dataset"""
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    
    def generator():
        for i in range(min(num_samples, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]
    
    return generator

def quantize_model(model_path, output_path, data_path, num_calibration_samples=100):
    """Quantize model to INT8"""
    
    print("="*60)
    print("MODEL QUANTIZATION")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite with quantization
    print("\nConverting to TFLite with INT8 quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(data_path, num_calibration_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert
    tflite_quantized_model = converter.convert()
    
    # Save quantized model
    with open(output_path, 'wb') as f:
        f.write(tflite_quantized_model)
    
    # Get model sizes
    original_size = os.path.getsize(model_path) / 1024
    quantized_size = len(tflite_quantized_model) / 1024
    
    print("\n" + "="*60)
    print("QUANTIZATION COMPLETED")
    print("="*60)
    print(f"Original model size: {original_size:.2f} KB")
    print(f"Quantized model size: {quantized_size:.2f} KB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    print(f"\n✅ Quantized model saved to {output_path}")
    print("="*60)
    
    return tflite_quantized_model

def evaluate_quantized_model(tflite_model, data_path):
    """Evaluate quantized model accuracy"""
    
    print("\n" + "="*60)
    print("EVALUATING QUANTIZED MODEL")
    print("="*60)
    
    # Load test data
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    # Initialize interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    # Run inference
    predictions = []
    for i in range(len(X_test)):
        # Quantize input
        input_data = X_test[i:i+1].astype(np.float32)
        input_data = input_data / input_scale + input_zero_point
        input_data = input_data.astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Dequantize output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output_data[0][0])
    
    predictions = np.array(predictions)
    y_pred = (predictions > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\nQuantized model accuracy: {accuracy:.4f}")
    print("="*60)
    
    return accuracy

def main(args):
    """Main quantization function"""
    
    # Quantize model
    tflite_model = quantize_model(
        args.model_path,
        args.output_path,
        args.data_dir,
        args.num_calibration_samples
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_quantized_model(tflite_model, args.data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize model to INT8 for MAX78000')
    parser.add_argument('--model_path', type=str, default='../models/mobilenetv2_base.h5',
                        help='Path to trained model')
    parser.add_argument('--output_path', type=str, default='../models/mobilenetv2_quantized.tflite',
                        help='Path to save quantized model')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--num_calibration_samples', type=int, default=100,
                        help='Number of samples for calibration (default: 100)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate quantized model on test set')
    
    args = parser.parse_args()
    main(args)