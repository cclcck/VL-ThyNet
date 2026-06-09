# VL-ThyNet: An Interpretable Multimodal Analytical Framework for Malignancy Risk Assessment of C-TIRADS 4 Thyroid Nodules

This repository implements the multimodal deep learning framework VL-ThyNet from the paper:
"VL-ThyNet: An interpretable multimodal analytical framework integrating visual foundation models and vision-language reasoning for malignancy risk assessment of C-TIRADS 4 thyroid nodules".
VL-ThyNet integrates a vision foundation model (DINOv2) with a ResNet adapter to extract fine-grained spatial and global features from paired transverse and longitudinal ultrasound images. By combining these dual-view visual features with late-fusion clinical characteristics, the model provides a highly accurate and interpretable binary classification (Benign vs. Malignant) specifically targeted at intermediate-risk C-TIRADS 4 thyroid nodules.

# Main Dependencies

- Python 3.8+
- PyTorch 1.13+
- CUDA 11.7 (if using GPU)

Full dependencies are listed in `requirements.txt`.

# Dataset Preparation

The raw data should include:

1. Paired thyroid nodule ultrasound images (`.png` or `.jpg`) containing both transverse and longitudinal cross-sections.

2. A clinical information table (`.csv`) containing the following essential fields:
   - Image Paths: image1_path, image2_path
   - Basic Features: age, gender

3. Place all images into `data/images/`  and the clinical configuration table into `data/clinical_data.csv` .