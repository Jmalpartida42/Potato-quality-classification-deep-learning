# Potato Quality Grading Pipeline: Segmentation & Classification 🥔

This project implements a high-precision computer vision pipeline for automated potato grading. The system combines a **U-Net** architecture for segmentation and size measurement with a **ResNet50** for surface defect detection.

## 📋 Grading Logic
The system evaluates each tuber based on the following business rules:
* **First Class (Premium):** No surface damage AND size above the threshold.
* **Second Class:** No surface damage AND size below the threshold.
* **Third Class:** Presence of surface damage (greening or rot), regardless of size.

---

## 🏗️ Pipeline Architecture
Processing is performed in two sequential stages to maximize accuracy:

### 1. Segmentation & Sizing (U-Net)
* **Backbone:** MobileNetV2 (Transfer Learning from ImageNet).
* **Function:** Generates the segmentation mask to calculate real dimensions and eliminate background noise.
* **Optimization:** Adam with Weight Decay.
* **Results:**
    * Cross-Validation (Average): **97.54% IoU** (~70 epochs).
    * **Test Set IoU: 98.01%** (70 epochs).

### 2. Quality Classification (ResNet50)
* **Architecture:** ResNet50 with Fine-tuning and ImageNet weights.
* **Process:** 1. Training of dense layers (Adam + L2 Regularization).
    2. Deep block fine-tuning (Adam with Weight Decay + L2).
* **Results:**
    * Cross-Validation: **98.13% Recall / 98.74% Precision** (~80 epochs).
    * **Test Set: 100% Precision / 100% Recall** (80 epochs).

---

## 🛡️ Robustness and Generalization (Model Stability)
To ensure the model is reliable and avoids data "memorization," the following techniques were applied:

* **Cross-Validation (K-Fold):** Consistently high results across all cross-validation folds demonstrate the model's excellent **generalization capability** on new data.
* **Overfitting Analysis:** Learning curves (Training vs. Validation Loss/Accuracy) show harmonic convergence. The minimal gap between training and validation metrics confirms that the model **does not suffer from overfitting**, thanks to the strategic use of:
    * L2 Regularization.
    * Weight Decay.
    * Controlled Fine-tuning.

Figure 1. Fold1 Results: Train vs. Validation (U-Net Model)
<img width="778" height="580" alt="image" src="https://github.com/user-attachments/assets/6ba8b8dd-d12f-4438-ba76-3dc39d6b99a5" />
<img width="778" height="580" alt="image" src="https://github.com/user-attachments/assets/f3a80ce7-bdba-4cc4-9b2f-951977c6327d" />

Figure 2. Fold1 Results: Train vs. Validation (ResNet50 Model)
<img width="1858" height="697" alt="image" src="https://github.com/user-attachments/assets/35a00d73-a9af-436b-90b9-ca77bc61c1e8" />

---

## 📊 Dataset & Training Strategy
A balanced dataset of **400 images** was used:
* **200 High-Quality Images:** No defects.
* **200 Defective Images:** 100 green potatoes (solanine) and 100 rotten potatoes.

**Data Distribution:**
* **80% Train-Validation:** Training using **Cross-Validation**.
* **20% Test Set:** 80 images reserved for final evaluation. The 100% accuracy results are strictly based on this independent group.

---

## 📸 Image Requirements
To ensure accuracy, input images must meet the following criteria:
1. **Solid Black Background:** Crucial for perfect segmentation.
2. **Uniform Lighting:** Avoid excessive glare.
3. **Input Example:**
   <img width="512" height="576" alt="0_27" src="https://github.com/user-attachments/assets/d5bc32ee-3e6b-4217-94eb-eede5f7bded4" />
