# Prodigy-InfoTech--Task-03
Prodigy InfoTech- Task-03


# **🐱🐶 Cats vs Dogs Image Classification Using HOG, PCA, and SVM**

## **📌 Project Overview**

This project focuses on building an intelligent image classification system that can automatically identify whether an image contains a **Cat** or a **Dog**. The model uses a combination of **Histogram of Oriented Gradients (HOG)** for feature extraction, **Principal Component Analysis (PCA)** for dimensionality reduction, and **Support Vector Machine (SVM)** for classification.

The objective of this project is to demonstrate how traditional Machine Learning techniques can effectively solve image classification problems without using Deep Learning models.

---

# **🎯 Project Objectives**

### **Main Goals**

* Develop an image classification system for Cats and Dogs.
* Extract meaningful image features using HOG descriptors.
* Reduce high-dimensional feature space using PCA.
* Train an SVM classifier for accurate prediction.
* Evaluate model performance using Accuracy, Classification Report, and Confusion Matrix.
* Visualize predictions and feature variance retention.

---

# **📂 Dataset Information**

### **Dataset Used**

**Microsoft Cats and Dogs Dataset (PetImages)**

### **Classes**

| Class | Label |
| ----- | ----- |
| Cat   | 0     |
| Dog   | 1     |

### **Images Used**

* Maximum Cat Images: **4000**
* Maximum Dog Images: **4000**
* Total Images: **8000**

### **Image Preprocessing**

* Converted to Grayscale
* Resized to **128 × 128**
* Normalized Pixel Values
* HOG Feature Extraction Applied

---

# **🛠 Technologies and Libraries Used**

### **Programming Language**

* Python

### **Libraries**

* NumPy
* OpenCV (cv2)
* Matplotlib
* Seaborn
* Scikit-Learn
* Scikit-Image
* tqdm

---

# **🔄 Project Workflow**

## **Step 1: Data Collection**

Images are loaded from:

```text
PetImages/
│
├── Cat/
└── Dog/
```

The program reads image files from both folders.

---

## **Step 2: Image Preprocessing**

Each image undergoes:

### **Operations Performed**

* Grayscale Conversion
* Image Resizing (128×128)
* Pixel Normalization

### **Benefits**

* Reduces computational complexity.
* Makes all images uniform in size.
* Improves training efficiency.

---

## **Step 3: HOG Feature Extraction**

### **What is HOG?**

Histogram of Oriented Gradients (HOG) extracts edge and shape information from images.

### **Parameters Used**

```python
orientations = 9
pixels_per_cell = (8,8)
cells_per_block = (2,2)
block_norm = 'L2-Hys'
```

### **Advantages**

* Captures object shape.
* Robust against lighting variations.
* Excellent for object recognition.

---

## **Step 4: Feature Combination**

The project combines:

### **Raw Pixel Features**

```text
128 × 128 = 16,384 Features
```

### **HOG Features**

Additional edge-based descriptors.

### **Combined Feature Vector**

```text
Raw Pixels + HOG Features
```

This creates a richer representation of image content.

---

## **Step 5: Train-Test Split**

Dataset split:

| Dataset  | Percentage |
| -------- | ---------- |
| Training | 80%        |
| Testing  | 20%        |

### **Benefits**

* Prevents overfitting.
* Enables fair model evaluation.

---

## **Step 6: Feature Scaling**

### **Technique Used**

```python
StandardScaler()
```

### **Purpose**

* Mean = 0
* Standard Deviation = 1

This ensures all features contribute equally during training.

---

## **Step 7: PCA Dimensionality Reduction**

### **Why PCA?**

The combined feature vector contains thousands of features.

PCA reduces dimensionality while preserving important information.

### **Configuration**

```python
PCA Components = 350
```

### **Benefits**

* Faster training
* Lower memory usage
* Reduced noise
* Better generalization

---

# **📊 PCA Explained Variance Analysis**

### **Observations**

* Variance gradually increases as components increase.
* At **350 PCA Components**, approximately **76% variance** is retained.
* The red dashed line indicates the **95% variance threshold**.
* More PCA components would be required to reach 95%.

### **Interpretation**

Although 95% variance is not achieved, 350 components provide a good balance between:

* Computational efficiency
* Memory consumption
* Classification performance

---

# **🤖 SVM Model Training**

### **Algorithm Used**

```python
Support Vector Machine (SVM)
```

### **Kernel**

```python
RBF (Radial Basis Function)
```

### **Parameters**

```python
C = 10
gamma = scale
```

### **Why SVM?**

* Excellent for high-dimensional data.
* Works well with image features.
* Effective decision boundaries.
* Strong generalization capability.

---

# **📈 Model Performance**

## **Confusion Matrix Analysis**

### **Confusion Matrix Results**

| Actual Class | Predicted Cat | Predicted Dog |
| ------------ | ------------- | ------------- |
| Cat          | 280           | 120           |
| Dog          | 96            | 304           |

### **Interpretation**

#### **Correct Predictions**

* Cats correctly identified = **280**
* Dogs correctly identified = **304**

#### **Incorrect Predictions**

* Cats classified as Dogs = **120**
* Dogs classified as Cats = **96**

### **Total Samples**

```text
280 + 120 + 96 + 304 = 800
```

### **Accuracy Calculation**

```text
Accuracy =
(TP + TN) / Total Samples

= (280 + 304) / 800

= 584 / 800

= 73%
```

# **✅ Final Accuracy = 73%**

---

# **📋 Performance Summary**

| Metric              | Value |
| ------------------- | ----- |
| Accuracy            | 73%   |
| Correct Predictions | 584   |
| Wrong Predictions   | 216   |
| Test Samples        | 800   |

---

# **🔍 Sample Prediction Analysis**

### **Correct Predictions (Green)**

Examples where:

```text
True Label = Predicted Label
```

The model successfully recognized:

* Dog images
* Cat images

### **Wrong Predictions (Red)**

Examples where:

```text
True Label ≠ Predicted Label
```

Common reasons:

* Similar body shapes
* Low image quality
* Unusual poses
* Background noise
* Occlusion

---

# **📌 Key Findings**

### **Strengths**

✅ Successfully classifies Cats and Dogs

✅ HOG captures useful edge information

✅ PCA reduces feature dimensionality

✅ SVM handles high-dimensional data effectively

✅ Achieved good classification performance

---

### **Limitations**

❌ Accuracy can still be improved

❌ PCA retains only ~76% variance

❌ Sensitive to image quality

❌ Some Cat images resemble Dogs and vice versa

---

# **🚀 Future Improvements**

### **Possible Enhancements**

* Increase PCA components
* Hyperparameter tuning using GridSearchCV
* Use Color Images instead of Grayscale
* Apply Data Augmentation
* Increase dataset size
* Use CNN-based Deep Learning models
* Implement Transfer Learning with:

  * VGG16
  * ResNet50
  * MobileNet
  * EfficientNet

Expected accuracy can improve from **73% to 90%+** using modern deep learning techniques.

---

# **🏆 Project Conclusion**

This project successfully demonstrates a complete Machine Learning pipeline for image classification using **HOG + PCA + SVM**. The model processes Cat and Dog images, extracts meaningful visual features, reduces dimensionality, and classifies images with an overall accuracy of approximately **73%**.

The project highlights the effectiveness of traditional Machine Learning techniques in computer vision tasks and provides a strong foundation for future improvements using advanced Deep Learning architectures.

## **Final Result**

### **🐱🐶 Cats vs Dogs Classifier**

### **Algorithm: HOG + PCA + SVM**

### **Accuracy Achieved: 73%**

### **Status: Successfully Implemented and Evaluated ✅**
