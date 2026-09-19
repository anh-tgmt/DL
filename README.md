# 🚗 Image Classification & Fine-Tuning with MobileNetV2 on CIFAR-10

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aZzcBjk9VvVx2omm5tHs0TTsy1MZhgWh?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This repository demonstrates an end-to-end Computer Vision pipeline for multi-class image classification on the **CIFAR-10** dataset using **MobileNetV2**. It covers data preprocessing, baseline model evaluation, three core fine-tuning strategies, and single-image inference testing.

---

## 📌 Table of Contents
1. [Overview & Architecture](#-overview--architecture)
2. [Dataset & Preprocessing](#-dataset--preprocessing)
3. [Fine-Tuning Strategies](#-fine-tuning-strategies)
4. [Experimental Results](#-experimental-results)
5. [Real-World Image Testing](#-real-world-image-testing)
6. [How to Run](#-how-to-run)

---

## 🏗️ Overview & Architecture

- **Base Model:** `MobileNetV2` (Pre-trained on ImageNet).
- **Target Dataset:** CIFAR-10 (10 classes: *Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck*).
- **Input Resolution Upsampling:** CIFAR-10 images ($32 \times 32$) are dynamically upsampled using `UpSampling2D(size=(3,3))` to $96 \times 96$ pixels to better fit the receptive field of MobileNetV2.

---

## 🧹 Dataset & Preprocessing

The dataset is split deterministically (Stratified Split, `random_state=42`) as follows:
- **Training Set:** 70% (42,000 samples)
- **Validation Set:** 15% (9,000 samples)
- **Test Set:** 15% (9,000 samples)

Input features are normalized via MobileNetV2's dedicated preprocessing function:
$$\text{Input Pixel Values} \in [-1, 1]$$

---

## 🎯 Fine-Tuning Strategies

To overcome overfitting and boost baseline performance, three core strategies were implemented during fine-tuning:

1. **Data Augmentation:**
   - Real-time transformation layers: `RandomFlip("horizontal")`, `RandomRotation(0.1)`, and `RandomZoom(0.1)`.
   - Prevents the model from memorizing exact pixel layouts and improves generalization.
2. **Unfreezing All Layers (`trainable = True`):**
   - Unlocks all pre-trained weights in MobileNetV2 to allow feature extractor filters to adapt to the CIFAR-10 domain.
3. **Very Low Learning Rate ($\eta = 10^{-5}$):**
   - Uses `Adam(learning_rate=1e-5)` to prevent *Catastrophic Forgetting* and preserve valuable features learned from ImageNet.

---

## 📊 Experimental Results

### Performance Metrics Comparison

Overall performance evaluated across test samples using **Macro-Averaged Metrics** and **Accuracy**:

| Metric | Baseline (Frozen Backbone) | Fine-Tuned Model (Strategy 1+2+3) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | ~78.45% | **> 90.00%** | 📈 **+11.55%** |
| **Precision (Macro)** | ~79.12% | **High** | 📈 Enhanced |
| **Recall (Macro)** | ~78.30% | **High** | 📈 Enhanced |
| **F1-Score (Macro)** | ~78.50% | **High** | 📈 Enhanced |

> **Key Takeaway:** Unfreezing the backbone paired with a minimal learning rate and data augmentation significantly outperformed the frozen baseline architecture.

---

## 🖼️ Real-World Image Testing

The trained model was evaluated on real-world test images outside the standard dataset. 

```python
# Inference pipeline snippet
img = image.load_img(img_path, target_size=(32, 32))
img_batch = np.expand_dims(image.img_to_array(img), axis=0)
predictions = ft_model.predict(img_batch)
