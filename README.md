 Brain Tumor Classification using Multi-Modal MRI (ViT + SSL)

 Project Overview

This project focuses on brain tumor classification using multi-modal MRI images (T1, T2, FLAIR).
A Vision Transformer (ViT) combined with self-supervised learning (SSL) is used to improve feature learning and classification accuracy.

---

 Objective

* To classify brain tumors accurately from MRI scans
* To utilize multi-modal MRI data for better representation
* To reduce dependency on labeled data using self-supervised learning

---

 Dataset

* Source: Kaggle (Brain Tumor MRI Dataset)
* Modalities: T1, T2, FLAIR
* Classes: Glioma, Meningioma, Pituitary, No Tumor

---

 Methodology

1. Data preprocessing (resizing, normalization, augmentation)
2. Multi-modal MRI fusion (T1 + T2 + FLAIR)
3. Self-supervised learning using reconstruction
4. Vision Transformer (ViT) for feature extraction
5. Fine-tuning for classification

---

 Model

* Vision Transformer (ViT)
* Self-Supervised Learning (SSL)
* Reconstruction-based learning (MSE loss)

---

 Results

* Accuracy: **99.08%**
* Improved performance compared to baseline models

---

 How to Run

1. Clone the repository
2. Install required libraries
3. Open the `.ipynb` file in Jupyter Notebook / VS Code
4. Run all cells

---

 Future Improvements

* Deploy as web application
* Use larger medical datasets
* Experiment with hybrid CNN-Transformer models

---

 Author

SUBASRI U

