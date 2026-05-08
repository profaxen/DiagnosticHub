# Medical Image Disease Classification: Pneumonia Detection

## Project Overview
This project implements an end-to-end deep learning pipeline for classifying pneumonia from chest X-ray images. It compares a custom baseline CNN with an advanced transfer learning model (ResNet50V2) and includes model explainability using Grad-CAM.

## Dataset
- **Name**: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Selection Reason**: Highly relevant medical application, manageable size for demonstration, and standard benchmarking dataset.
- **Classes**: 2 (Normal, Pneumonia).
- **Format**: RGB (converted from grayscale), PNG/JPEG.

## Folder Structure
```
project/ 
├── data/               # Raw and processed data
├── notebooks/          # Exploration and experimentation
├── src/                # Modular Python scripts
│   ├── data_loading.py
│   ├── train_baseline.py
│   ├── train_advanced.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explainability.py
├── models/             # Saved model weights (.h5)
├── outputs/            # Generated plots and reports
├── README.md
├── requirements.txt
└── main.py             # Entry point for training
```

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download Dataset**:
   Download from Kaggle and place in `data/raw/` so that the path `data/raw/chest_xray/` exists with folders `train`, `val`, and `test`.
3. **Train Models**:
   ```bash
   python main.py --model baseline --epochs 10
   python main.py --model advanced --epochs 10
   ```
4. **Inference**:
   Use `src/predict.py` or the main notebook for individual image predictions.

## Key Features
- **Data Augmentation**: Robust preprocessing to prevent overfitting.
- **Class Balancing**: Uses class weights to handle dataset imbalance.
- **Transfer Learning**: Leverages ResNet50V2 for state-of-the-art performance.
- **Explainability**: Grad-CAM visualizations to identify regions of interest.

## Results Summary
- **Baseline CNN**: Faster training, lower accuracy (~85-88%).
- **Advanced Model**: Slower training (initially), significantly higher precision and recall (~94-96%).

## Disclaimer
This project is for educational and research purposes only. It is not intended for clinical use.
