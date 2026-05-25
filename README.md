# Employee Performance Prediction

This project analyzes employee performance data and provides a simple web-based prediction application.

## Project Summary

- `employee.ipynb` contains the full exploratory data analysis (EDA), statistical testing, preprocessing, and model training workflow.
- `app.py` is a Flask application that loads or trains a prediction model, accepts input through a web form, and returns a performance prediction.
- `templates/index.html` is the web UI template for the prediction form.
- `Employee_Perfomance.xls` is the dataset used for analysis and model training.

## What the project does

- Reads employee performance data from an Excel dataset.
- Performs EDA with visualizations such as distribution plots, boxplots, correlation heatmaps, and department-wise analysis.
- Runs statistical tests (ANOVA and Chi-square) to identify relationships between features and performance ratings.
- Encodes categorical variables and trains classification models to predict `PerformanceRating`.
- Provides a Flask-based prediction service where users can submit employee details and receive a predicted performance category.

## Key Components

- `employee.ipynb`
  - Data loading and inspection
  - EDA for numerical and categorical features
  - Correlation analysis and statistics
  - Preprocessing and encoding
  - Model training and evaluation with Random Forest, SVC, KNN, and Decision Tree

- `app.py`
  - Loads a saved model bundle if available
  - If `model.pkl` is missing, trains a fallback model from `Employee_Perfomance.xls`
  - Encodes form inputs and predicts employee performance
  - Renders results through the web interface

- `templates/index.html`
  - Contains the form for entering employee attributes
  - Displays prediction results on the same page

## Running the app

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install xlrd
   ```
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open the app in a browser:
   ```text
   http://127.0.0.1:5000/
   ```

## Notes

- The project uses label encoding for department and job role features.
- The Flask app is designed to work with the same feature set as the notebook.
- If the trained model file is missing, the app trains a fallback model from the dataset automatically.

## Improvements

Potential next steps:
- Save the final model and encoder bundle to `model.pkl` for consistent deployment.
- Add input validation and clearer error handling in the Flask UI.
- Improve model evaluation and hyperparameter tuning in the notebook.
