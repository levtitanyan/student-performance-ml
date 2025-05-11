#  Student Performance ML Project

This project applies machine learning techniques to analyze and predict student academic outcomes.  
It demonstrates key ML skills in regression, classification, feature engineering, evaluation, and modular Python design.

---

## Dataset Information

- Source: UCI Machine Learning Repository  
  [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- Description:  
  The dataset contains student records including demographic, social, and academic attributes.  
  Target variables:
  - final_grade (0–20): used for regression
  - passed (binary): 1 if final_grade >= 12, else 0 — used for classification

---

## Project Structure

student-performance-ml/
├── data/                  # Raw and cleaned datasets
│   ├── data-raw.csv
│   ├── data-corrupted.csv # With null values added
│   ├── data-cleaned.csv
|
├── notebooks/             # Step-by-step Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
|
├── scripts/               # Modular Python scripts
│   ├── data_loader.py
│   ├── eda_plots.py
│   ├── models.py
│   ├── preprocessing.py
│   └── saver.py
|
├── outputs/               # Generated outputs
│   ├── plots/             # All saved figures
│   └── reports/           # Evaluation result CSVs
|
├── requirements.txt       # Project dependencies
└── README.md              # This file

---

##  How to Run

1. Clone the repository
   
   git clone https://github.com/levtitanyan/student-performance-ml.git
   cd student-performance-ml
   
2. Install dependencies
   
   pip install -r requirements.txt
   
3. Launch notebooks
   Open the notebooks in order:
   - 01_eda.ipynb
   - 02_preprocessing.ipynb
   - 03_modeling_regression.ipynb
   - 04_modeling_classification.ipynb

---

## Modeling Overview

### Regression Targets:
- final_grade

Models Used:
- Linear Regression
- Ridge
- Lasso
- ElasticNet
- Random Forest

### Classification Targets:
- passed (binary from final_grade ≥ 12)

Models Used:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes

---

## Output Artifacts

- 📊 All plots saved to outputs/plots/
- 📑 Evaluation reports saved to outputs/reports/
- 💡 Trained models saved to models/ (e.g., ridge_model.pkl, `logreg_model.pkl`)

---

##  Results

###  Best Regression Model: Lasso Regression
- Test R²: 0.808  

###  Best Classification Model: Decision Tree Classifier
- Test Accuracy: 91.5%, F1 Score: 0.893

---

## Authors

- Arstvik Avetisyan [GitHub Profile](https://github.com/Artsvik9)
- Levon Titanyan [GitHub Profile](https://github.com/levtitanyan)