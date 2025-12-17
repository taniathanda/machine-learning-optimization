# 🎬 IMDB Movie Rating Prediction & Optimization using Deep Learning

## 📌 Project Overview
This project builds an end-to-end machine learning and deep learning pipeline to predict **movie quality (Good vs Bad)** using **IMDB movie descriptions and metadata**.  
It integrates baseline machine learning, deep learning (LSTM), Bayesian hyperparameter optimization, and model explainability techniques.

---

## 🎯 Objectives
- Predict movie quality based on IMDB data
- Compare traditional machine learning with deep learning models
- Optimize deep learning models using Bayesian Optimization (Optuna)
- Explain model predictions using SHAP
- Build an interpretable and reusable ML pipeline

---

## 🗂️ Dataset
- Source: IMDB movie dataset
- Features:
  - Movie description (text)
  - Rating
  - Votes
  - Duration
  - Release year
  - Genre
- Target:
  - Binary classification:
    - Good movie (rating ≥ threshold)
    - Bad movie (rating < threshold)

---

## 🧹 Data Preprocessing
- Text cleaning and normalization
- Tokenization and sequence padding
- Genre extraction and encoding
- Numerical feature scaling
- Missing value handling
- Binary label creation

---

## 🧠 Models Implemented

### 1️⃣ Baseline Model
- TF-IDF vectorization
- Logistic Regression
- Purpose:
  - Establish baseline performance
  - Provide model comparison with deep learning

---

### 2️⃣ Deep Learning Model
- LSTM (Long Short-Term Memory)
- Architecture:
  - Embedding layer
  - LSTM layer
  - Dropout layers
  - Dense output layer (Sigmoid)
- Designed to capture sequential patterns in text

---

### 3️⃣ Optimized Deep Learning Model
- Hyperparameter optimization using Optuna
- Tuned parameters:
  - Embedding dimension
  - Number of LSTM units
  - Dropout rate
  - Learning rate
  - Optimizer type
- Objective:
  - Maximize validation accuracy

---

## ⚙️ Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Validation accuracy and loss curves

---

## 🔍 Model Explainability
- SHAP (SHapley Additive Explanations) used for interpretability
- Surrogate model built for deep learning explanations
- Insights include:
  - Impact of numerical features (votes, duration, year)
  - Important keywords influencing predictions

---

## 📈 Results Summary
- Optimized LSTM model achieved higher accuracy than baseline model
- Bayesian optimization significantly improved validation performance
- SHAP analysis provided transparency into model decisions

---

## 🛠️ Tech Stack
- Programming Language:
  - Python
- Libraries and Frameworks:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - TensorFlow / Keras
  - Optuna
  - SHAP
  - NLTK

---

## 📂 Project Workflow
- Load and explore IMDB dataset
- Perform exploratory data analysis (EDA)
- Preprocess text and numerical features
- Train baseline TF-IDF + Logistic Regression model
- Build LSTM deep learning model
- Optimize model using Optuna
- Evaluate best model on test data
- Explain predictions using SHAP
- Save trained models and outputs

---

## ▶️ How to Run

### Clone the repository
```bash
git clone <[your-repository-url](https://github.com/taniathanda/machine-learning-optimization.git)>
cd <repository-folder>
