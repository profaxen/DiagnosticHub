# Final Project Report: Medical Image Disease Classification (Pneumonia)

## 1. Abstract
This project presents a comparative study of two deep learning architectures for the automated detection of Pneumonia from chest X-ray images. By leveraging the Chest X-ray (Pneumonia) dataset, we developed a baseline CNN and an advanced transfer learning model based on ResNet50V2. Our findings demonstrate that transfer learning significantly improves recall—a critical metric in medical diagnostics.

## 2. Problem Statement
Pneumonia is a life-threatening condition that requires timely diagnosis. Manual interpretation of X-rays is time-consuming and subject to human error. This project aims to build a robust, interpretable AI system to assist clinicians in rapid screening.

## 3. Methodology
### 3.1 Data Preparation
- **Dataset**: 5,856 X-ray images.
- **Preprocessing**: Images resized to 224x224 and normalized.
- **Augmentation**: Applied rotations and flips to improve generalization.
- **Handling Imbalance**: Calculated class weights to give higher importance to the minority class (Normal).

### 3.2 Model Architectures
- **Baseline**: 3-layer CNN with Dropout and Max-Pooling.
- **Advanced**: ResNet50V2 backbone with a custom classification head. Transfer learning was used by freezing early layers and fine-tuning the later stages.

## 4. Results and Comparison
| Metric | Baseline CNN | Advanced (ResNet50V2) |
|--------|--------------|-----------------------|
| Accuracy | ~87% | ~95% |
| Precision | ~84% | ~93% |
| Recall | ~91% | ~97% |
| F1-Score | ~87% | ~95% |

### Insights:
- The advanced model showed superior performance in recall, which is vital to avoid missing positive cases (False Negatives).
- Baseline model was prone to overfitting without extensive regularization.

## 5. Model Explainability (Grad-CAM)
Using Grad-CAM, we visualized that the model primarily focuses on the lung regions where opacities (signs of pneumonia) are present. This "sanity check" is crucial for building trust with medical professionals.

## 6. Limitations and Future Scope
### Limitations:
- Dataset represents a specific demographic; performance might vary on different populations.
- Binary classification does not account for other lung pathologies (e.g., COVID-19, Tuberculosis).

### Future Improvements:
- Multi-class classification for multiple lung diseases.
- Integration of patient metadata (age, symptoms) for multi-modal diagnosis.
- Deployment via a web-based dashboard for real-time inference.

## 7. Conclusion
The advanced ResNet50V2 model is a viable tool for pneumonia screening, offering high sensitivity and interpretability. This project demonstrates the power of transfer learning in medical imaging where data can be scarce or imbalanced.
