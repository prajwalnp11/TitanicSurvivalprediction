# TitanicSurvivalprediction
# Titanic Survival Prediction

A machine learning web application that predicts whether a Titanic passenger survived based on passenger details such as class, sex, age, ticket fare, family members aboard, and port of embarkation.

## Project Overview

This project uses Python and scikit-learn to train a classification model on the Titanic dataset. A Flask web application provides an HTML form where users enter passenger details and receive a predicted survival outcome with an estimated probability.

## Features

- Predict passenger survival using machine learning.
- Compare Logistic Regression and Random Forest models.
- Handle missing numerical and categorical values.
- Scale numerical features and encode categorical features.
- Engineer additional features: `FamilySize` and `IsAlone`.
- Display predictions through a web interface built with HTML and CSS.
- Show estimated survival probabilities.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- HTML and CSS
- Git and GitHub

## Dataset

The dataset is available from the Kaggle Titanic competition:

https://www.kaggle.com/competitions/titanic/data

The training dataset contains passenger features and the `Survived` target column.

- `0` — Did not survive
- `1` — Survived

The project uses features including `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

Additional engineered features:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1` when the passenger travelled alone; otherwise `0`.

## Machine Learning Workflow

1. Load the Titanic training and test datasets.
2. Perform exploratory data analysis.
3. Handle missing values using imputation.
4. Scale numerical features using `StandardScaler`.
5. Encode categorical features using `OneHotEncoder`.
6. Train and compare classification models.
7. Select the best model based on validation performance.
8. Save the trained model as `titanic_model.pkl`.
9. Load the model into the Flask application.
10. Accept passenger details and display predictions.

## Project Structure

```text
Titanic-Survival-Prediction/
├── app.py
├── README.md
├── requirements.txt
├── titanic_model.pkl
├── templates/
│   └── index.html
└── static/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/prajwalnp11/TitanicSurvivalprediction.git
cd TitanicSurvivalprediction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

Open the following URL in your browser:

http://127.0.0.1:5000/

Make sure `titanic_model.pkl` is present in the project directory before starting the application.

## Example Usage

1. Open the web application.
2. Enter passenger details, including passenger class, sex, age, ticket fare, family members aboard, and embarkation port.
3. Click **Predict Survival**.
4. View the predicted outcome and estimated survival probability.

## Limitations

- Predictions are based on historical Titanic passenger data.
- Missing values are handled using statistical imputation.
- Performance depends on the selected features, model, and validation methodology.
- Estimated probabilities are not guarantees of actual outcomes.
- The application requires compatible Python and scikit-learn versions for loading the saved model.

## Future Improvements

- Add cross-validation and hyperparameter tuning.
- Evaluate models using precision, recall, F1-score, and a confusion matrix.
- Improve the user interface and input validation.
- Deploy the application to a public hosting platform.
- Add automated testing and continuous integration.

## Disclaimer

This project is intended for educational purposes and demonstrates an end-to-end machine learning workflow. Predictions are estimates, not historical facts or guarantees.

## Author

Prajwal N P

GitHub: https://github.com/prajwalnp11