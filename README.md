# Autonomous Medical Image Triage Agent

**Multi-Label Chest X-Ray Classification using Deep Learning**

A CNN-based pathology detection system that analyzes chest X-rays to detect multiple pathological conditions and prioritize cases by urgency.

## Authors
- Vishal Kumar
- Deepashree Srinivasa Rao Rannore

**Course:** DS 5500 - Data Science Capstone  
**Institution:** Northeastern University

---

## Problem Statement

Radiologists are overwhelmed with massive scan volumes, and critical cases often wait hours in the queue alongside routine cases. Our system analyzes chest X-rays to detect pathological conditions and prioritize life-threatening cases for immediate attention.

---

## Dataset

**CheXpert** (Stanford ML Group)
- 224,316 chest radiographs from 65,240 patients
- 14 pathology labels with uncertainty annotations
- Source: [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)

---

## Model Architecture

| Model | Parameters | Best AUC |
|-------|------------|----------|
| DenseNet-121 | 7M | **0.816** |
| EfficientNet-B0 | 5M | 0.812 |

### Target Pathologies
- Cardiomegaly
- Pneumonia
- Pneumothorax
- Edema
- Pleural Effusion

---

## Results

### Per-Class AUC (DenseNet-121)

| Pathology | AUC |
|-----------|-----|
| Pleural Effusion | 0.845 |
| Edema | 0.821 |
| Cardiomegaly | 0.802 |
| Pneumothorax | 0.755 |
| Pneumonia | 0.677 |

---

## Project Structure

```
├── notebooks/
│   └── chexpert_classification.ipynb    # Main training notebook
├── outputs/
│   ├── densenet121_best.pth             # Trained model weights
│   ├── efficientnet_b0_best.pth
│   ├── training_curves.png
│   ├── per_class_auc.png
│   └── model_comparison.csv
└── README.md
```

---

## Setup & Usage

### 1. Clone Repository
```bash
git clone https://github.com/zavisk/AutonomousMedicalImage_TriageAgent.git
```

### 2. Open in Google Colab
- Upload notebook to Colab
- Select GPU runtime (A100 recommended)

### 3. Configure Kaggle API
```python
kaggle_credentials = {
    "username": "YOUR_USERNAME",
    "key": "YOUR_API_KEY"
}
```

### 4. Run All Cells
Training takes ~30-40 minutes on A100 GPU.

---

## Key Features

- **Transfer Learning**: ImageNet pretrained weights
- **U-Ones Policy**: Uncertain labels treated as positive
- **Data Augmentation**: Random flip, rotation, color jitter
- **Multi-Label Classification**: BCEWithLogitsLoss

---

## Technologies

- PyTorch
- torchvision
- scikit-learn
- pandas
- matplotlib

---

## Future Work

- [ ] Add LR scheduler for improved convergence
- [ ] Implement class weights for imbalanced data
- [ ] Build RAG pipeline for evidence-based recommendations
- [ ] Develop agentic workflow for autonomous triage
- [ ] Create Streamlit demo interface

---

## References

1. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning.
2. Irvin, J., et al. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels.
