# Heart Disease Prediction App

This project is a **machine learning–based web application** that predicts the risk of heart disease using patient clinical data.  
The model is trained using **Gaussian Naive Bayes** and deployed using **Streamlit**.

## Features

- Predicts **heart disease risk (Yes / No)**
- Displays **probability score**
- Uses a **robust preprocessing pipeline** with:
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
  - Passthrough for binary features
- Clean and interactive **Streamlit UI**

## Machine Learning Model

- **Algorithm**: Gaussian Naive Bayes  
- **Evaluation Metrics**:
  - Accuracy: **~87.5%**
  - F1 Score: **~0.88**

### Models Compared
| Model | Accuracy | F1 Score |
|-----|--------|---------|
| Logistic Regression | 0.8696 | 0.8812 |
| KNN | 0.8641 | 0.8792 |
| **Naive Bayes** | **0.8750** | **0.8844** |
| Decision Tree | 0.7717 | 0.7941 |
| SVM (RBF) | 0.8587 | 0.8713 |

## Dataset Features

| Feature | Description |
|------|------------|
| Age | Age of patient |
| Sex | Male / Female |
| ChestPainType | ATA, NAP, ASY, TA |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar (0/1) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate |
| ExerciseAngina | Exercise-induced angina (0/1) |
| Oldpeak | ST depression |
| ST_Slope | Up, Flat, Down |

## Preprocessing Pipeline

Implemented using `ColumnTransformer`:

- **Numeric Columns** → `StandardScaler`
- **Categorical Columns** → `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Binary Columns** → `passthrough`

This ensures **consistent preprocessing during training and deployment**.
