# Image-Classification
Flower Image Classification 
# 🌸 Flower Classification with DenseNet121

This project classifies flower images into 5 categories using transfer learning with DenseNet121.

## 📂 Dataset
- The dataset used in this project is the **"Flowers - Five Classes"** dataset from [Kaggle](https://www.kaggle.com/datasets/lara311/flowers-five-classes?resource=download).
- Classes: `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`
- The dataset is split into:
  - `train/`
  - `val/`
  - `test/`

## 🧼 Preprocessing and Augmentation

### 🖼️ Image Preprocessing
- **Image Size**: Resized to `240 × 240` pixels
- **Normalization**: Pixel values rescaled to the range `[0, 1]`

### 🔁 Data Augmentation (via Albumentations)
Applied only to training data to improve generalization

## 🧠 Model Architecture

leveraged **Transfer Learning** using the pre-trained **DenseNet121** as the feature extractor, followed by a custom classification head tailored for 5-class flower classification.

### 🔧 Architecture Overview

- **Base Model**: `DenseNet121` from Keras Applications
  - Pre-trained on **ImageNet**
  - `include_top=False`: removed the final classification layer
  - **Frozen** during training to preserve learned features


## 🏋️‍♂️ Training & Evaluation

- The model was trained for **20 epochs** using the **Adam optimizer** and **Sparse Categorical Crossentropy** as the loss function.
- A **batch size of 32** was used to ensure efficient GPU utilization.
- **Training and validation accuracies** were monitored to ensure proper convergence and generalization.
- To avoid or minimize overfitting, **Dropout layers (rate = 0.3)** were added after the dense layers.
- Evaluation was done using multiple metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- The model achieved approximately **~89% accuracy**, showing robust performance across all five flower categories.
