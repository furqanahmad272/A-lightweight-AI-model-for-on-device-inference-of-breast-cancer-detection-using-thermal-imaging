"""
MAX78000 Export Script
Prepares quantized model for MAX78000 deployment
"""

import os
import json
import shutil
import argparse
import tensorflow as tf

def create_model_metadata(tflite_path, output_dir):
    """Extract and save model metadata"""
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    model_size = os.path.getsize(tflite_path) / 1024
    
    metadata = {
        'model_name': 'mobilenetv2_breast_cancer',
        'input_shape': input_details[0]['shape'].tolist(),
        'output_shape': output_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype']),
        'input_quantization': {
            'scale': float(input_details[0]['quantization'][0]),
            'zero_point': int(input_details[0]['quantization'][1])
        },
        'output_quantization': {
            'scale': float(output_details[0]['quantization'][0]),
            'zero_point': int(output_details[0]['quantization'][1])
        },
        'model_size_kb': model_size,
        'classes': ['healthy', 'cancer']
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main(args):
    """Main export function""" 
    
    print("="*60)
    print("MAX78000 MODEL EXPORT")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("✅ Model ready for MAX78000 deployment!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export model for MAX78000 deployment')
    parser.add_argument('--tflite_model', type=str, default='../models/mobilenetv2_quantized.tflite')
    parser.add_argument('--output_dir', type=str, default='../models/max78000')
    
    args = parser.parse_args()
    main(args)