# Credit Risk Analysis

## Overview
The Credit Risk Analysis project applies **Machine Learning (ML)** techniques to classify loan applications using the German Credit Dataset. With 1,000 loan applicant records and 20 feature variables, this project aims to extract crucial attributes influencing credit risk and build an efficient **classification model** to predict creditworthiness.

## Key Features

- **Data Preprocessing**: Cleans and prepares the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Identifies patterns and correlations in credit risk factors.
- **Feature Engineering**: Selects key attributes to enhance classification accuracy.
- **Machine Learning Models**: Implements classifiers such as Logistic Regression, Decision Trees, and Random Forest.
- **Model Evaluation**: Assesses model performance using accuracy, precision, recall, and confusion matrices.

## Project Files

### 1. `german_credit_data.csv`
This dataset contains information about loan applicants, including their credit status and associated attributes.

Key Fields:
- `Age`: Age of the applicant.
- `Credit Amount`: Loan amount requested.
- `Duration`: Loan repayment duration in months.
- `Job`: Type of employment.
- `Housing`: Housing status (own, rent, free).
- `Risk`: Classification label (Good or Bad credit risk).

### 2. `main.py`
This script processes the dataset, trains classification models, and evaluates their performance.

**Key Components:**

#### Data Loading & Cleaning
- Reads and preprocesses data from `german_credit_data.csv`.
- Handles missing values and normalizes numerical features.

#### Feature Engineering
- Encodes categorical variables.
- Performs feature selection for optimal model performance.

#### Model Training & Evaluation
- Splits data into training and test sets.
- Trains classifiers such as Logistic Regression, Decision Trees, and Random Forest.
- Evaluates models using accuracy, F1-score, and confusion matrices.

### 3. `prediction.py`
This script allows users to input new applicant data and predict credit risk using the trained model.

**Functionality:**
- Accepts applicant details as input.
- Preprocesses input data.
- Applies the trained model for prediction.

### 4. `requirements.txt`
Lists required dependencies for running the project.

## Example Code
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('german_credit_data.csv')

# Preprocessing
X = data.drop(columns=['Risk'])
y = data['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

## How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Main Script
```bash
python main.py
```

### Step 3: Predict New Data
```bash
python prediction.py
```

## Future Enhancements
- **Deep Learning Models**: Implement neural networks for improved accuracy.
- **Explainable AI (XAI)**: Provide insights into credit approval decisions.
- **Automated Feature Selection**: Optimize feature selection using advanced techniques.
- **Real-Time Prediction API**: Deploy a live API for credit risk predictions.

## Conclusion
The Credit Risk Analysis project leverages **Machine Learning (ML)** techniques to classify loan applicants based on financial data. By implementing robust **classification models**, this project enhances credit risk assessment, enabling better financial decision-making.


**Happy Analyzing 🚀!**
