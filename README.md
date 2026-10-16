# 🚦 Traffic Sign Recognition Using CNN in Deep Learning

An end-to-end Computer Vision Deep Learning application that recognizes and classifies road traffic signs with high confidence using a trained Convolutional Neural Network (CNN) built in TensorFlow/Keras and served through an interactive Streamlit web dashboard.

---

## 📌 Project Overview & Objectives

In smart city infrastructure and Autonomous Driving Systems (ADAS), automated traffic sign recognition is fundamental for vehicle safety, speed compliance, and collision prevention.

This application fulfills all objectives outlined in the project abstract:
1. **Image Input:** Accepts user-uploaded road sign photos in JPG, JPEG, PNG, BMP, or WebP format.
2. **Standardized Preprocessing:** Resizes input images to uniform 64×64 pixels and normalizes pixel values to [0.0, 1.0] using bilinear interpolation.
3. **Hierarchical Feature Extraction:** Applies 3 convolutional blocks (32, 64, 128 filters) with Batch Normalization and MaxPooling2D to capture edges, textures, and geometric road sign motifs.
4. **Softmax Multiclass Classification:** Fully connected layers (128 neurons, Dropout 0.4) output calibrated probabilities across 58 traffic sign categories.
5. **Transparency & Confidence:** Displays top-3 predicted candidates with probabilities and confidence ratings.
6. **Verified Evaluation Dashboard:** Real accuracy, loss curves, confusion matrices, and precision/recall evaluated against held-out validation and test sets.

---

## 🏗️ Neural Network Architecture

```
Layer (type)                     Output Shape          Param #    Details
===================================================================================
InputLayer                       (None, 64, 64, 3)     0          RGB Image
Conv2D (3x3, same)               (None, 64, 64, 32)    896        ReLU activation
BatchNormalization               (None, 64, 64, 32)    128        Momentum 0.9
MaxPooling2D (2x2)               (None, 32, 32, 32)    0          Stride 2
Conv2D (3x3, same)               (None, 32, 32, 64)    18,496     ReLU activation
BatchNormalization               (None, 32, 32, 64)    256        Momentum 0.9
MaxPooling2D (2x2)               (None, 16, 16, 64)    0          Stride 2
Conv2D (3x3, same)               (None, 16, 16, 128)   73,856     ReLU activation
BatchNormalization               (None, 16, 16, 128)   512        Momentum 0.9
MaxPooling2D (2x2)               (None, 8, 8, 128)     0          Stride 2
Flatten                          (None, 8192)          0          Flattens feature map
Dense                            (None, 128)           1,048,704  ReLU activation
Dropout (0.4)                    (None, 128)           0          Regularization
Dense (Output)                   (None, 58)            7,482      Softmax activation
===================================================================================
Total Parameters: 1,150,330 (4.39 MB)
```

---

## 📂 Project Directory Structure

```
Traffic sign Recognition project/
│
├── app.py                            # Streamlit entry point & Landing Page
├── train_model.py                    # Script to train CNN & export artifacts
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── .streamlit/
│   └── config.toml                   # Custom Navy/Slate UI theme configuration
│
├── pages/
│   ├── 1_Traffic_Sign_Recognition.py # Interactive prediction & inference page
│   ├── 2_Model_Performance.py        # Accuracy, loss curves & confusion matrix
│   ├── 3_How_It_Works.py             # Beginner-friendly pipeline explanation
│   └── 4_Model_Information.py        # Architecture specs & 58 class index
│
├── utils/
│   ├── __init__.py
│   ├── model_loader.py               # st.cache_resource model & metadata loader
│   ├── predictor.py                  # Image preprocessing & inference logic
│   └── class_mapping.py              # Class ID to Sign Label helper
│
├── model/
│   ├── traffic_sign_cnn.keras        # Saved trained Keras model
│   ├── class_mapping.json            # Class IDs and labels mapping
│   ├── metrics.json                  # Real evaluated training and test metrics
│   └── training_history.json         # Epoch-by-epoch loss & accuracy history
│
└── archive/                          # Dataset (excluded from Git)
    ├── labels.csv                    # 58 class definitions
    └── traffic_Data/
        ├── DATA/                     # Labeled training images by folder
        └── TEST/                     # Held-out test images
```

---

## 🚀 Setup & Local Execution (Windows)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- VS Code, Antigravity IDE, or Windows Terminal.

### 2. Install Dependencies
Open your PowerShell or Command Prompt in the project folder and run:
```powershell
pip install -r requirements.txt
```

### 3. Model Training & Artifact Generation (Optional / Automated)
If the trained model artifacts in `model/` are already present, you can skip to Step 4.  
To retrain or re-evaluate the model from scratch:
```powershell
python train_model.py
```

### 4. Launch the Streamlit Web Application
Run the Streamlit server:
```powershell
streamlit run app.py
```
Your default web browser will automatically open:
```
http://localhost:8501
```

---

## 🌐 Online Deployment (Streamlit Community Cloud)

1. Push your project files to a **GitHub repository** (ensure `.gitignore` excludes the large `archive/` folder; the `model/` directory with `traffic_sign_cnn.keras` (4.4 MB) is small and comfortably fits within GitHub's 100 MB limit).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"**, select your repository, branch (`main`), and set the main file path to:
   ```
   app.py
   ```
4. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and launch your live application.

---

## 💡 Model Accuracy vs. Prediction Confidence

A critical distinction emphasized in this project:
- **Model Accuracy:** Evaluated across the entire held-out test dataset ($$\frac{\text{Correct}}{\text{Total}} \times 100$$).
- **Prediction Confidence:** The single-sample probability produced by the Softmax output layer for one specific uploaded image.
