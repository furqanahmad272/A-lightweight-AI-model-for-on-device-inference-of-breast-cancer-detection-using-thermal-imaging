# Edge AI: Ultra-Lightweight MobileNetV2 for Breast Cancer Detection on Wearables

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An ultra-lightweight deep learning model for breast cancer detection using thermal imaging data, optimized for deployment on ultra-low-power AI accelerators like the **MAX78000** MCU in wearable devices.

## 🎯 Project Overview

This project implements a highly efficient **MobileNetV2** model designed specifically for edge AI deployment on wearable devices. The model analyzes thermal imaging data to detect breast cancer, with optimizations including:

- **Ultra-lightweight architecture** (MobileNetV2 with α=0.35)
- **INT8 quantization** for minimal memory footprint
- **Export pipeline** to TFLite and MAX78000 C header formats
- **Ready for deployment** on ultra-low-power AI accelerators

## 📁 Project Structure

```
.
├── data/                           # Dataset storage
│   ├── raw/                       # Raw thermal images
│   ├── processed/                 # Preprocessed data
│   └── README.md                  # Dataset instructions
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training_quantization.ipynb
│   └── 03_model_export.ipynb
├── scripts/                        # Python scripts
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Model training
│   ├── quantize.py                # Model quantization
│   └── export_for_max78000.py     # MAX78000 export
├── models/                         # Saved models
│   ├── mobilenetv2_base.h5
│   ├── mobilenetv2_quantized.tflite
│   └── max78000/                  # MAX78000 deployment files
├── docs/                           # Documentation
│   ├── MAX78000_DEPLOYMENT.md     # Deployment guide
│   ├── DATASET.md                 # Dataset information
│   └── MODEL_ARCHITECTURE.md      # Model details
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/furqanahmad272/A-lightweight-AI-model-for-on-device-inference-of-breast-cancer-detection-using-thermal-imaging.git
cd A-lightweight-AI-model-for-on-device-inference-of-breast-cancer-detection-using-thermal-imaging
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Follow instructions in `data/README.md` to download and prepare the thermal imaging dataset.

### 4. Run Notebooks

Execute notebooks in order:
1. **Data Preprocessing**: `notebooks/01_data_preprocessing.ipynb`
2. **Model Training & Quantization**: `notebooks/02_model_training_quantization.ipynb`
3. **Model Export**: `notebooks/03_model_export.ipynb`

## 📊 Dataset

**Recommended Dataset**: [DMR-IR Breast Thermography Dataset](https://visual.ic.uff.br/dmi/)

- **Type**: Thermal infrared images
- **Classes**: Healthy vs. Cancer
- **Preprocessing**: Resize to 96×96 or 128×128, normalize to [0,1]

See `docs/DATASET.md` for detailed dataset information and alternatives.

## 🧠 Model Architecture

- **Base**: MobileNetV2 (ImageNet architecture)
- **Width Multiplier (α)**: 0.35 (ultra-lightweight)
- **Input Size**: 96×96×3 or 128×128×3
- **Parameters**: ~50K (after pruning)
- **Output**: Binary classification (Healthy / Cancer)

### Optimizations:
- ✅ Depthwise separable convolutions
- ✅ INT8 quantization
- ✅ Quantization-Aware Training (QAT)
- ✅ Channel pruning

See `docs/MODEL_ARCHITECTURE.md` for technical details.

## 🔧 Usage

### Training the Model

```bash
python scripts/train.py --data_dir data/processed --epochs 50 --batch_size 32
```

### Quantizing the Model

```bash
python scripts/quantize.py --model_path models/mobilenetv2_base.h5 --output_path models/mobilenetv2_quantized.tflite
```

### Exporting for MAX78000

```bash
python scripts/export_for_max78000.py --tflite_model models/mobilenetv2_quantized.tflite --output_dir models/max78000
```

## 📱 Deployment on MAX78000

The model is optimized for deployment on the **Analog Devices MAX78000** ultra-low-power AI accelerator.

### Deployment Steps:
1. Export quantized TFLite model
2. Convert to MAX78000 format using `ai8x-synthesis`
3. Generate C header files
4. Integrate into firmware

See `docs/MAX78000_DEPLOYMENT.md` for detailed deployment instructions.

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Model Size | ~50 KB |
| Parameters | ~50K |
| Inference Time (MAX78000) | <100 ms |
| Power Consumption | <1 mW |
| Accuracy | ~92% (on validation set) |

*Note: Actual performance depends on dataset and hardware configuration*

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.9.0** / **Keras**
- **TensorFlow Lite** (Quantization)
- **OpenCV** (Image preprocessing)
- **NumPy** / **Pandas**
- **Matplotlib** / **Seaborn** (Visualization)
- **Jupyter Notebook**
- **ai8x-synthesis** (MAX78000 SDK)

## 📖 Documentation

- [Dataset Information](docs/DATASET.md)
- [Model Architecture Details](docs/MODEL_ARCHITECTURE.md)
- [MAX78000 Deployment Guide](docs/MAX78000_DEPLOYMENT.md)

## 🔬 Research & References

1. **MobileNetV2**: [Paper](https://arxiv.org/abs/1801.04381)
2. **MAX78000**: [Datasheet](https://www.analog.com/en/products/max78000.html)
3. **Thermal Imaging for Breast Cancer**: [Research](https://visual.ic.uff.br/dmi/))

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- Analog Devices for MAX78000 documentation
- TensorFlow Lite team for quantization tools
- Research community for thermal imaging datasets

---

**Note**: This project is for research and educational purposes. Medical diagnosis should always be performed by qualified healthcare professionals.
