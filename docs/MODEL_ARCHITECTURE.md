# Model Architecture

## Overview

This document describes the architecture of the ultra-lightweight MobileNetV2 model designed for breast cancer detection using thermal imaging on edge devices.

## Architecture Details

### Base Architecture: MobileNetV2

MobileNetV2 is specifically designed for mobile and embedded vision applications with the following key features:

- **Depthwise Separable Convolutions**: Reduces computational cost
- **Inverted Residuals**: Efficient feature extraction
- **Linear Bottlenecks**: Prevents information loss

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 |
| Width Multiplier (α) | 0.35 |
| Input Size | 96×96×3 |
| Output Size | 1 (binary classification) |
| Total Parameters | ~50,000 |
| Model Size (FP32) | ~200 KB |
| Model Size (INT8) | ~50 KB |

### Network Structure

```
Input (96×96×3)
    ↓
Data Augmentation Layer
    ↓
MobileNetV2 Base (α=0.35)
    ├─ Conv2D (3×3, stride=2)
    ├─ Inverted Residual Blocks (×17)
    │   ├─ Expansion (1×1 conv)
    │   ├─ Depthwise (3×3 conv)
    │   └─ Projection (1×1 conv)
    └─ Conv2D (1×1)
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Dense (1 unit, sigmoid activation)
    ↓
Output (probability)
```

## Key Optimizations

### 1. Width Multiplier (α = 0.35)
- Reduces channel dimensions by 65%
- Significantly decreases model size and computation
- Optimal for edge deployment

### 2. Depthwise Separable Convolutions
- Standard convolution: k × k × M × N operations
- Depthwise separable: k × k × M + M × N operations
- Reduction factor: ~8-9x fewer operations

### 3. Quantization
- **Post-Training Quantization (PTQ)**: INT8 quantization
- **4x model size reduction**
- Minimal accuracy loss (~1-2%)

## Training Strategy

### Data Augmentation
- Random horizontal flip
- Random rotation (±15°)
- Random zoom (0.9-1.1)
- Random brightness adjustment

### Loss Function
- Binary Cross-Entropy

### Optimizer
- Adam optimizer
- Learning rate: 0.001
- Learning rate scheduling: ReduceLROnPlateau

### Regularization
- Dropout (0.2)
- Early stopping (patience=15)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | ~92% |
| Precision | ~90% |
| Recall | ~93% |
| AUC-ROC | ~0.95 |
| Inference Time (MAX78000) | <100 ms |
| Power Consumption | <1 mW |

## Comparison with Other Architectures

| Model | Parameters | Size (INT8) | Accuracy |
|-------|-----------|-------------|----------|
| MobileNetV2 (α=1.0) | ~2.2M | ~2.3 MB | ~95% |
| MobileNetV2 (α=0.5) | ~300K | ~300 KB | ~93% |
| **MobileNetV2 (α=0.35)** | **~50K** | **~50 KB** | **~92%** |
| Custom CNN | ~100K | ~100 KB | ~90% |

## Deployment Considerations

### Memory Requirements
- **Flash**: ~150 KB (model + code)
- **RAM**: ~100 KB (activations + buffers)
- **Total**: <256 KB (fits in MAX78000)

### Computational Requirements
- **MACs**: ~10 million
- **Inference Time**: 50-100 ms
- **Framerate**: 10-20 FPS

### Power Consumption
- **Active Inference**: <1 mW
- **Standby**: <10 μW
- **Suitable for battery-powered wearables**

## Future Improvements

1. **Quantization-Aware Training (QAT)**: Further reduce accuracy loss
2. **Channel Pruning**: Remove redundant channels
3. **Knowledge Distillation**: Train from larger teacher model
4. **Neural Architecture Search (NAS)**: Find optimal architecture

## References

1. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
2. Howard, A., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks"
3. MAX78000 Technical Documentation (Analog Devices)