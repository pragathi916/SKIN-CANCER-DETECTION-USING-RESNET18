# Skin Cancer Detection using Deep Learning (HAM10000 Dataset)

This project is a skin cancer detection pipeline built using PyTorch and the HAM10000 dataset. It classifies dermatoscopic images into cancerous or non-cancerous lesions using a deep convolutional neural network.

## 🔍 Dataset
- **Name:** HAM10000 (Human Against Machine with 10000 training images)
- **Source:** [Kaggle - kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Classes:** 7 skin lesion types (grouped into binary labels: `cancer` or `non-cancer`)
- **License:** CC BY-NC-SA 4.0

## 📁 Project Structure
```
├── HAM10000_metadata.csv         # Metadata file
├── HAM10000_images/             # All merged images
├── skin-cancer-mnist-ham10000.zip
├── best_cancer_model.pth        # Saved best model
├── skin_cancer_detection.ipynb  # Main training notebook
```

## 🧠 Model Architecture
- **Backbone:** ResNet18 (pretrained on ImageNet)
- **Modifications:**
  - Final FC layer replaced with a single-node output layer
  - Added dropout for regularization
  - Sigmoid activation for binary classification

## ⚙️ Training Pipeline
1. Upload `kaggle.json` to authenticate Kaggle API
2. Download and extract HAM10000 dataset
3. Merge images into one folder
4. Prepare DataFrame and assign binary labels:
   - `mel`, `bkl`, `bcc`, `akiec` => **Cancer (1)**
   - `nv`, `df`, `vasc` => **Non-Cancer (0)**
5. Split into train, val, and test (stratified)
6. Apply transforms (resize, normalize)
7. Train ResNet18 using BCEWithLogitsLoss
8. Evaluate with Accuracy, AUC, F1 Score
9. Save best model (`best_cancer_model.pth`)

## 📊 Evaluation Metrics
- Accuracy
- AUC (ROC)
- F1 Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

## 📦 Dependencies
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- PIL

## 🚀 Run it on Google Colab
Make sure to:
- Upload `kaggle.json`
- Run the notebook cells in order
- Upload any custom image to test after training completes

## 📌 Notes
- This is a binary classification task grouped from 7 classes.
- The model uses transfer learning (ResNet18 pretrained on ImageNet).
- Ensure a balanced dataset for better performance (handled via stratified split).


