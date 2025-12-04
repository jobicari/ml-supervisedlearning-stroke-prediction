# Stroke Prediction Using Machine Learning  
**Supervised Learning Final Project – Master's in AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project explores predictive modeling for stroke risk using a real-world healthcare dataset.  
The focus is on **careful EDA**, **model comparison**, **hyperparameter tuning**, **threshold optimization**, and **interpretability** through **feature importance analysis**.

---

## Dataset Attribution & Usage Notice (APA Style)

This project uses the publicly available *Healthcare Dataset — Stroke Data* from Kaggle.

Sorian̄o, F. (2020). *Stroke Prediction Dataset* [Data set]. Kaggle.  
https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

This dataset is used strictly for educational purposes as part of a supervised learning project.  
If reused, you must cite the dataset author as shown above.

---


## Project Structure
├── ml-stroke-prediction.ipynb     # Full analysis, modeling, and commentary
└──data
    └── healthcare-dataset-stroke-data.csv
├── custom.css                    # Notebook styling (optional)
├── environment.yml               # Reproducible environment (conda)
└── README.md

---

## Project Goals

1. Conduct exploratory data analysis (EDA) to understand feature distributions and data quality.  
2. Handle **high class imbalance** (~5% stroke prevalence).  
3. Train and compare multiple ML models:
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting (XGBoost & SKLearn)  
   - Support Vector Classifier (RBF kernel)  
4. Perform **hyperparameter optimization** using GridSearchCV.  
5. Evaluate performance using:
   - ROC–AUC  
   - Precision, Recall, F1  
   - Confusion matrices  
6. Apply **threshold tuning** using Fβ scores (β = 1, 2, 3) to emphasize recall.  
7. Interpret model behavior via **feature importance** (XGBoost & Random Forest).  
8. Summarize findings and produce actionable conclusions.

---

## Dataset Overview

The dataset contains demographic and health-related features:

- **Numerical:** `age`, `avg_glucose_level`, `bmi`  
- **Categorategorical:** `gender`, `hypertension`, `heart_disease`, `ever_married`,  
  `work_type`, `residence_type`, `smoking_status`

### **Key Observations**
- **Severe class imbalance:** ~95% non-stroke vs. ~5% stroke.  
- Certain features (e.g., hypertension, heart disease) show distributions aligned with stroke prevalence.  
- Some categories (e.g., *Other* for gender, *Never worked*) are extremely rare.  
- Numerical variables show mild separation between stroke vs. non-stroke groups.

---

## Statistical Tests Included

- **Mann–Whitney U Test** for numerical feature differences by stroke outcome  
- **Chi-Squared Independence Tests** for categorical features  
- Results confirm **statistically significant differences** across several key predictors.

---

## Models Trained & Tuned

Each model is placed in a preprocessing pipeline combining:

- **ColumnTransformer**  
  - Standard scaling (numerical)  
  - One-hot encoding (categorical)

### **Models Evaluated**
- Logistic Regression  
- Random Forest  
- Gradient Boosting (sklearn)  
- Support Vector Classifier  
- XGBoost Classifier  

### **Tuning Strategy**
- GridSearchCV with ROC–AUC scoring  
- Adjustment of key hyperparameters based on model family  
- Use of `class_weight='balanced'` or `scale_pos_weight` for imbalanced data

---

## Threshold Optimization

Because stroke detection demands **high recall**, fixed 0.50 thresholds underperform.

We apply **Fβ optimization** for β ∈ {1, 2, 3}:

- **β = 1:** balanced precision–recall  
- **β = 2:** recall-heavy  
- **β = 3:** strongly recall-dominant  

### Outcome
Threshold tuning significantly improved recall across all models while maintaining reasonable precision.

---

## Feature Importance Analysis

Using **XGBoost (best-performing model)**:

Top contributing features include:

- **Age**  
- **Average glucose level**  
- **Hypertension**  
- **Heart disease**  
- One-hot encoded categories for work type and smoking status also contribute meaningfully.

This step enhances interpretability and links ML outcomes to clinical insights.

---

## Model Comparison

A compact comparison table summarizes:

- ROC–AUC (CV)  
- ROC–AUC (test)  
- Precision, Recall, F1  
- Threshold-tuned results for β = 3  

### Key Finding
Across all three rounds of tuning:

**XGBoost and Random Forest consistently perform best**,  
with **SVC competitive** but slightly below tree ensembles in recall after tuning.

---

## Final Conclusions

1. **Class imbalance is severe**, making ROC–AUC a more stable metric than accuracy.  
2. Tree-based models (RF, XGBoost) handle heterogeneous tabular data more effectively.  
3. **Threshold tuning is crucial**—default cutoff of 0.50 misses too many positive strokes.  
4. Best-performing configuration:
   - **XGBoost at β = 3 threshold**
   - Achieves strong recall while maintaining usable precision  
5. Feature importance aligns well with established medical risk factors.

---

## Installation & Reproducibility

### **Using Conda (recommended)**

```bash
conda env create -f environment.yaml
conda activate stroke-prediction