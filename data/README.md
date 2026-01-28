# Dataset Information

This directory contains thermal imaging datasets for breast cancer detection.

## 📊 Recommended Datasets

### 1. DMR-IR Dataset (Primary)

**Source**: [Visual Computing Lab - UFF](https://visual.ic.uff.br/dmi/)

- **Type**: Infrared thermal images
- **Patients**: 45 patients
- **Images**: ~200 thermal images
- **Classes**: Healthy, Sick (with cancer)
- **Format**: PNG/JPEG
- **Resolution**: Various (will be resized to 96×96 or 128×128)

#### Download Instructions:
1. Visit: https://visual.ic.uff.br/dmi/
2. Request access to DMR-IR dataset
3. Download and extract to `data/raw/dmr_ir/`

### 2. Alternative Datasets

#### Kaggle - Breast Thermal Images
- **Source**: https://www.kaggle.com/datasets/sfreis/visual-dmr
- **Size**: ~200 images
- **Classes**: Normal, Cancer

#### Database for Mastology Research (DMR)
- **Source**: Research institutions
- **Contact**: Request through academic channels

## 📁 Directory Structure

```
data/
├── raw/                    # Raw downloaded datasets
│   ├── dmr_ir/            # DMR-IR dataset
│   │   ├── healthy/
│   │   └── cancer/
│   └── other/             # Other datasets
├── processed/              # Preprocessed data
│   ├── train/
│   │   ├── healthy/
│   │   └── cancer/
│   ├── val/
│   │   ├── healthy/
│   │   └── cancer/
│   └── test/
│       ├── healthy/
│       └── cancer/
└── README.md              # This file
```

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. **Resize**: 96×96 or 128×128 pixels
2. **Normalization**: Pixel values to [0, 1]
3. **Augmentation**:
   - Horizontal flip
   - Rotation (±15°)
   - Brightness adjustment
   - Zoom (0.9-1.1)
4. **Split**: 70% train, 15% validation, 15% test

## 📝 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | ~200 |
| Training Set | ~140 |
| Validation Set | ~30 |
| Test Set | ~30 |
| Image Size | 96×96 or 128×128 |
| Color Channels | 3 (RGB) |

## ⚠️ Important Notes

1. **Data Privacy**: Ensure compliance with medical data regulations (HIPAA, GDPR)
2. **Ethical Use**: This data is for research purposes only
3. **Class Imbalance**: May need to apply class weighting or oversampling
4. **Preprocessing**: Always preprocess raw data before training

## 🔗 Additional Resources

- [Thermal Imaging in Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6316596/)
- [Breast Cancer Detection Research](https://arxiv.org/abs/2104.08289)

## 📧 Contact for Dataset Access

If you need help accessing datasets, please contact:
- Email: furqanahmad272@github.com
- Institutional access may be required for some datasets