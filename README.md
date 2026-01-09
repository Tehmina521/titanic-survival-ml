# Titanic Survival Prediction

## Project Overview
This project predicts passenger survival on the Titanic using machine learning.  
The dataset contains demographic and travel information such as age, sex, class, fare, and family relationships.  
The goal is to build an end-to-end **ML pipeline** that demonstrates data preprocessing, feature engineering, model training, evaluation, and interpretation.

---

## Problem Statement
The Titanic disaster dataset is used to answer the question:  
**“Given a passenger’s information, can we predict whether they survived the disaster?”**

---

## Dataset
- **Source:** Kaggle Titanic dataset (`train.csv` and `test.csv`)  
- **Key Features:**
  - `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`  
- **Target:** `Survived` (0 = No, 1 = Yes)

---

## Project Workflow

### 1. Data Loading & Exploration
- Load training and test datasets.  
- Inspect missing values and data types.  
- Explore survival patterns by `Sex` and `Pclass`.

### 2. Feature Engineering
- Create new features:
  - `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`, `AgeBin`.  
- Handle missing values for `Age`, `Fare`, and `Embarked`.  
- Encode categorical features for ML.

### 3. Preprocessing
- **Numeric features:** Standard scaling.  
- **Categorical features:** One-hot encoding.  
- Combined preprocessing and modeling in a pipeline.

### 4. Train/Validation Split
- 80/20 split with stratification to preserve class distribution.  

### 5. Modeling
- Random Forest Classifier with tuned hyperparameters.  
- Class weights applied to address class imbalance.  

### 6. Threshold Tuning
- Adjusted probability threshold using F1-score optimization.  
- Ensures balanced detection of survivors and non-survivors.

### 7. Model Evaluation
- Metrics:
  - Accuracy, Precision, Recall, F1-score  
  - ROC-AUC  
- Visualizations:
  - Confusion Matrix  
  - ROC Curve  
  - Feature Importances

---

## Key Decisions & Reasoning
- **Feature Engineering:** Captures family size, social status, and normalized fare to improve predictive power.  
- **Random Forest Model:** Robust to mixed data types, non-linearities, and provides feature importance.  
- **Threshold Tuning:** Maximizes F1-score due to class imbalance.  
- **Pipeline:** Ensures reproducible and clean workflow.  

---

## Results
- **Validation Accuracy:** ~84%  
- **F1 Score:** ~0.79 for survivors  
- **Top Features:**
  - `Age`, `Title`, `Gender`

---

## Tech Stack
- **Language:** Python 3.x  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Model:** Random Forest Classifier  
- **Environment:** Jupyter Notebook  

---

## ▶️ How to Run

1. Clone this repository
    ```bash
   git clone https://github.com/Tehmina521/titanic-survival-ml.git
2. Navigate into the folder
   *cd titanic-survival-ml*
3. Install dependencies:
pip install -r requirements.txt
3. Open the notebook in 
**jupyter notebook**
4. Run the cells step by step to reproduce the analysis and results.
---

## 👩‍💻 Author

Tehmina Afzal
AI/ML Intern – SkillLink Nexus Ltd
  
