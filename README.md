# 🚢 Titanic Survival Prediction

Predicting Titanic passenger survival using **feature engineering**, **Random Forest**, and **threshold-tuned machine learning**. This project demonstrates a complete ML workflow from **data preprocessing** to **deployment-ready predictions**.

---

## Dataset
- **Source:** Kaggle Titanic Dataset (`train.csv`, `test.csv`)  
- **Target Variable:** `Survived` (0 = did not survive, 1 = survived)  
- **Key Features:**  
  - `Pclass` — passenger class (1, 2, 3)  
  - `Sex` — male/female  
  - `Age` — passenger age  
  - `Fare` — ticket fare  
  - `Embarked` — port of embarkation (C, Q, S)  
  - `FamilySize` — number of family members onboard  
  - `IsAlone` — traveling alone or not  
  - `FarePerPerson` — Fare divided by family size  
  - `Title` — social title (Mr, Mrs, Miss, Master, Rare)  
  - `AgeBin` — age categories (Child, Teen, Adult, MidAge, Senior)  

---

## Project Workflow

### 1. Data Exploration
- Checked missing values and types
- Analyzed survival rates by `Sex` and `Pclass`

### 2. Feature Engineering
- Created: `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`, `AgeBin`  
- Filled missing values using **train median/mode**  

### 3. Preprocessing
- Numeric features scaled  
- Categorical features encoded  
- Random Forest combined with preprocessing pipeline  

### 4. Train/Validation Split
- 80% train / 20% validation, stratified  

### 5. Model Training & Threshold Tuning
- Random Forest Classifier  
- Tuned probability threshold to **0.44** to maximize **F1-score for survivors**  

### 6. Evaluation
- Accuracy: ~82.7%  
- ROC-AUC: ~0.84  
- F1-score (Survived): 0.78  
- Confusion Matrix:

![Confusion Matrix](images/confusion_matrix.png)

- ROC Curve:

![ROC Curve](images/roc_curve.png)

- Feature Importance:

![Feature Importance](images/feature_importance.png)

---

## Performance Summary

| Metric | Default Threshold (0.5) | Tuned Threshold (0.44) |
|--------|------------------------|-----------------------|
| Accuracy | 0.8268 | 0.8268 |
| Recall (Survived) | 0.70 | 0.78 ✅ improved |
| F1-score (Survived) | 0.76 | 0.78 ✅ improved |
| ROC-AUC | 0.84 | 0.84 |

**Key Insights:**  
- Females and first-class passengers had higher survival rates  
- Small families survived better than alone passengers  
- Young boys (`Master`) and unmarried women (`Miss`) had higher survival probability  

---

## Single Passenger Prediction
```python
from titanic_model_pipeline import predict_single_passenger

sample = {
    'Pclass': 1, 'Sex':'female', 'Age':29, 'Fare':72.5,
    'Embarked':'C','FamilySize':1,'IsAlone':1,
    'FarePerPerson':72.5, 'Title':'Miss', 'AgeBin':'Adult'
}

result = predict_single_passenger(sample)
print(result)
