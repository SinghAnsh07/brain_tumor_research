# Brain Tumor Classification & Explainable AI (XAI) Research

This repository contains a comprehensive research project focused on multi-class brain tumor classification using deep learning architectures and attention mechanisms, combined with Explainable AI (XAI) analysis to validate clinical faithfulness.

---

## 📌 Project Overview
Brain tumors require highly accurate and rapid diagnosis. While deep learning models offer state-of-the-art accuracy, their "black-box" nature presents a barrier to clinical adoption. This project tackles both challenges by:
1. **Developing a robust deep learning framework** to classify brain MRI scans into four classes: **Glioma, Meningioma, Pituitary, and No Tumor**.
2. **Integrating and comparing attention modules** (Squeeze-and-Excitation, CBAM, and Hybrid) to improve classification accuracy and feature focus.
3. **Validating model explanations** using **Grad-CAM** visual heatmaps and verifying them quantitatively using the **ROAD (Remove and Debias)** explanation faithfulness metric.

---

## 🧪 Ablation Study Configuration
We systematically evaluated three base architectures across four attention configurations, resulting in a **12-model ablation study**:

*   **Base Architectures:**
    *   **ResNet50** (Residual learning framework)
    *   **DenseNet121** (Densely connected convolutional network)
    *   **EfficientNetB5** (Compound-scaled scaling framework)
*   **Attention Configurations:**
    *   **Baseline** (No Attention)
    *   **Squeeze-and-Excitation (SE)** (Channel-wise relationship modeling)
    *   **Convolutional Block Attention Module (CBAM)** (Spatial and Channel attention sequence)
    *   **Hybrid** (Combined Squeeze-and-Excitation + CBAM blocks)

---

## 📊 Evaluation & Research Results

### 1. Full Ablation Table (Local Run)
*Tested on Local Machine with input resolution `128x128` and Batch Size `16`:*

| Model ID | Base Architecture | Attention Module | Test Accuracy | Status / Winner |
| :--- | :--- | :--- | :---: | :--- |
| **A1** | EfficientNetB5 | None | 90.75% | |
| **A2** | EfficientNetB5 | SE | 92.31% | |
| **A3** | EfficientNetB5 | CBAM | 91.25% | |
| **A4** | EfficientNetB5 | Hybrid | 91.81% | |
| **B1** | ResNet50 | None | 82.38% | |
| **B2** | ResNet50 | SE | 85.44% | |
| **B3** | ResNet50 | CBAM | 78.25% | |
| **B4** | ResNet50 | Hybrid | 82.44% | |
| **C1** | DenseNet121 | None | 93.56% | |
| **C2** | **DenseNet121** | **SE** | **93.94%** | 🏆 **Local Winner** |
| **C3** | DenseNet121 | CBAM | 93.31% | |
| **C4** | DenseNet121 | Hybrid | 93.25% | |

---

### 2. High-Quality Fine-Tuning (Colab Run)
*Tested on Google Colab with input resolution `224x224` and Batch Size `32` for the top performing candidates:*

| Model ID | Base Architecture | Attention Module | Test Accuracy | Status / Winner |
| :--- | :--- | :--- | :---: | :--- |
| **C2** | **DenseNet121** | **SE** | **95.00%** | 🏆 **Overall Winner** |
| **C1** | DenseNet121 | None | 94.81% | |
| **A2** | EfficientNetB5 | SE | 93.50% | |

---

### 3. Classification Report: Best Model (C2 - DenseNet121 + SE, 224x224)
*Test Support: 1600 images total (400 per class)*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 1.00 | 0.84 | 0.91 | 400 |
| **Meningioma** | 0.92 | 0.96 | 0.94 | 400 |
| **No Tumor** | 0.93 | 1.00 | 0.96 | 400 |
| **Pituitary** | 0.96 | 1.00 | 0.98 | 400 |
| **Overall (Macro Avg)** | **0.98** | **0.98** | **0.98** | **1600** |

---

### 4. Explainable AI (XAI) Faithfulness - ROAD Scores
We computed the **ROAD (Remove and Debias)** score to evaluate the faithfulness of the Grad-CAM heatmaps. Higher ROAD scores mean that removing the features highlighted by Grad-CAM drops the model confidence the most, confirming the model actually relies on those clinically correct tumor boundaries.

*   **Mean ROAD Score:** **0.4205** (indicates highly faithful explanations)
*   **Per-Class ROAD Scores:**
    *   **Pituitary:** `0.6231` (Highest explainability alignment)
    *   **Meningioma:** `0.5822`
    *   **Glioma:** `0.3200` (Lowest score, consistent with being the most diffuse/difficult class)
    *   **No Tumor:** `0.1567` (Expected lower score as there is no localized tumor to focus on)

---

### 5. Visual Results

#### Grad-CAM Explainability — Best Model (DenseNet121 + SE)
The Grad-CAM heatmaps below show where the model focuses when classifying each tumor type. Red regions indicate high attention. The overlays confirm that the model correctly localizes pathological tumor regions across all four classes.

<p align="center">
  <img src="gradcam%20results/gradcam_publication.png" alt="Grad-CAM Explainability — DenseNet121 + SE Attention" width="800"/>
</p>

#### Grad-CAM Heatmap Grid — All Classes
<p align="center">
  <img src="gradcam%20results/gradcam_all_classes.png" alt="Grad-CAM Visualizations — All Classes" width="800"/>
</p>

#### ROAD Faithfulness Score — Per-Class Breakdown
The bar chart below summarizes the per-class ROAD (Remove and Debias) scores. Higher scores indicate that the Grad-CAM explanations faithfully reflect the features the model actually relies on for prediction.

<p align="center">
  <img src="gradcam%20results/road_scores.png" alt="ROAD Score per Class — DenseNet121 + SE" width="700"/>
</p>

---

## 💡 Key Research Findings
1. **Squeeze-and-Excitation (SE)** attention consistently improved test accuracy across **all three** base architectures.
2. **CBAM (Convolutional Block Attention Module)** was highly inconsistent and actually reduced performance for ResNet50 and DenseNet121 compared to their baselines.
3. **Hybrid Attention** (SE + CBAM) was never the top performer, meaning channel-wise attention (SE) alone is superior to compounding spatial and channel attention in this task.
4. **DenseNet121** proved to be the most resilient base model for medical imaging classification, significantly outperforming ResNet50 and EfficientNetB5.
5. **Clinical Alignment:** Grad-CAM heatmaps show that the model accurately highlights the pathological location of tumors (gliomas, meningiomas, and pituitary tumors) in the brain.

---

## 📁 Repository Structure
```directory
├── dataset/                         # Local MRI scans dataset (organized by train/test classes)
├── gradcam results/                 # Grad-CAM & ROAD XAI visual outputs
│   ├── gradcam_publication.png      # Publication-quality Grad-CAM heatmaps (all 4 classes)
│   ├── gradcam_all_classes.png      # Grad-CAM grid visualization
│   ├── road_scores.png              # Per-class ROAD faithfulness score bar chart
│   └── complete_results_summary.txt # Full numerical results summary
├── models/                          # Evaluation assets, charts, and metadata summaries
│   ├── Model A1..C4 confusion matrices
│   ├── eda_sample_images.png        # Exploratory Data Analysis sample distributions
│   └── results_summary.txt          # Model metrics summary log
├── .gitignore                       # Project gitignore file (ignores datasets & model weights)
├── densenet base.ipynb              # DenseNet121 Baseline training notebook
├── densenet se.ipynb                # DenseNet121 + SE Attention notebook
├── densenet cbam.ipynb              # DenseNet121 + CBAM Attention notebook
├── densenet hybrid.ipynb            # DenseNet121 + Hybrid Attention notebook
├── resnet base.ipynb                # ResNet50 Baseline notebook
├── resnet se.ipynb                  # ResNet50 + SE Attention notebook
├── resnet cbam.ipynb                # ResNet50 + CBAM Attention notebook
├── resnet hybrid.ipynb              # ResNet50 + Hybrid Attention notebook
├── efficientnet base se cbam.ipynb  # EfficientNetB5 Baseline + SE + CBAM notebook
├── efficientnet hybrid.ipynb        # EfficientNetB5 + Hybrid Attention notebook
├── top_three_models.ipynb           # Grad-CAM, ROAD XAI evaluation & final comparative analysis
└── README.md                        # Detailed research presentation and documentation
```

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (recommended for training)

### Dependencies Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SinghAnsh07/brain_tumor_research.git
   cd brain_tumor_research
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install tensorflow numpy matplotlib pandas opencv-python scikit-learn jupyter
   ```

---

## 🚀 How to Run the Research
1. Open the notebooks using Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run the base training notebooks (e.g., `densenet se.ipynb`) to train individual configurations. Trained weights will automatically save in Keras/H5 formats.
3. Open `top_three_models.ipynb` to evaluate trained models, generate Grad-CAM heatmaps, compute ROAD scores, and reproduce the comparative summary logs.
