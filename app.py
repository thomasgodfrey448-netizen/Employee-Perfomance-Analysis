from flask import Flask, render_template, request
import numpy as np
import pickle
import os

# CREATE FLASK APP
app = Flask(__name__)

# Default mapping for environment (string -> numeric), used if dataset not available
environment_dict = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

# ENCODING DICTIONARIES
emp_department_dict = {
    'Data Science': 0,
    'Development': 1,
    'Finance': 2,
    'Human Resources': 3,
    'Research & Development': 4,
    'Sales': 5
}

emp_jobrole_dict = {
    'Business Analyst': 0,
    'Data Scientist': 1,
    'Developer': 2,
    'Finance Manager': 3,
    'Healthcare Representative': 4,
    'Human Resources': 5,
    'Laboratory Technician': 6,
    'Manager': 7,
    'Manager R&D': 8,
    'Manufacturing Director': 9,
    'Research Director': 10,
    'Research Scientist': 11,
    'Sales Executive': 12,
    'Sales Representative': 13,
    'Senior Developer': 14,
    'Senior Manager R&D': 15,
    'Technical Architect': 16,
    'Technical Lead': 17
}

def ensure_model():
    """Load existing model bundle or train a fallback model from the notebook dataset.

    The saved bundle is a dict with keys: 'model', 'dept_encoder', 'jobrole_encoder', 'environment_dict'
    """
    if os.path.exists('model.pkl'):
        bundle = pickle.load(open('model.pkl', 'rb'))
        return bundle

    # Try loading an existing trained model with a different filename if present
    alt_names = ['employee_performance_rf_model.pkl', 'employee_performance_model.pkl']
    for alt in alt_names:
        if os.path.exists(alt):
            try:
                bundle = pickle.load(open(alt, 'rb'))
                return bundle
            except Exception:
                pass

    # Train a simple model from the dataset if model.pkl not present
    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_excel('Employee_Perfomance.xls', engine='xlrd')

        # Features (match the notebook)
        features = [
            'EmpEnvironmentSatisfaction',
            'EmpLastSalaryHikePercent',
            'ExperienceYearsInCurrentRole',
            'ExperienceYearsAtThisCompany',
            'YearsSinceLastPromotion',
            'YearsWithCurrManager',
            'EmpJobRole',
            'EmpDepartment'
        ]

        # Drop rows with missing values in required cols
        df = df.dropna(subset=features + ['PerformanceRating'])

        # Encode job role and department using LabelEncoder
        dept_enc = LabelEncoder()
        job_enc = LabelEncoder()
        df['EmpDepartment_enc'] = dept_enc.fit_transform(df['EmpDepartment'].astype(str))
        df['EmpJobRole_enc'] = job_enc.fit_transform(df['EmpJobRole'].astype(str))

        # Ensure environment is numeric; if strings, map using environment_dict
        if df['EmpEnvironmentSatisfaction'].dtype == object:
            df['EmpEnvironmentSatisfaction_enc'] = df['EmpEnvironmentSatisfaction'].map(environment_dict)
        else:
            df['EmpEnvironmentSatisfaction_enc'] = df['EmpEnvironmentSatisfaction']

        X = df[[
            'EmpDepartment_enc',
            'EmpJobRole_enc',
            'EmpLastSalaryHikePercent',
            'EmpEnvironmentSatisfaction_enc',
            'YearsSinceLastPromotion',
            'ExperienceYearsInCurrentRole',
            'ExperienceYearsAtThisCompany',
            'YearsWithCurrManager'
        ]].values
        y = df['PerformanceRating'].values

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        bundle = {
            'model': clf,
            'dept_encoder': dept_enc,
            'jobrole_encoder': job_enc,
            'environment_dict': environment_dict
        }

        # Save bundle for future use
        pickle.dump(bundle, open('model.pkl', 'wb'))
        return bundle
    except Exception:
        # If training fails, create a dummy majority-class predictor
        class DummyModel:
            def __init__(self, majority=3):
                self.majority = majority

            def predict(self, X):
                return np.array([self.majority] * len(X))

        dummy = DummyModel()
        bundle = {
            'model': dummy,
            'dept_encoder': None,
            'jobrole_encoder': None,
            'environment_dict': environment_dict
        }
        return bundle


bundle = ensure_model()
model = bundle['model']
dept_encoder = bundle.get('dept_encoder')
jobrole_encoder = bundle.get('jobrole_encoder')
env_mapping = bundle.get('environment_dict', environment_dict)

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# PREDICTION ROUTE
@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    department = request.form['EmpDepartment']
    jobrole = request.form['EmpJobRole']
    salary_hike = float(request.form['EmpLastSalaryHikePercent'])
    environment = request.form['EmpEnvironmentSatisfaction']
    years_promotion = float(request.form['YearsSinceLastPromotion'])
    current_role = float(request.form['ExperienceYearsInCurrentRole'])
    company_years = float(request.form['ExperienceYearsAtThisCompany'])
    manager_years = float(request.form['YearsWithCurrManager'])

    # Encode categorical values
    department_encoded = emp_department_dict[department]
    jobrole_encoded = emp_jobrole_dict[jobrole]
    environment_encoded = environment_dict[environment]

    # Feature array
    features = np.array([[
        department_encoded,
        jobrole_encoded,
        salary_hike,
        environment_encoded,
        years_promotion,
        current_role,
        company_years,
        manager_years
    ]])

    # Prediction
    prediction = model.predict(features)[0]

    # Convert prediction to label
    performance_dict = {
        2: 'Good Performance',
        3: 'Excellent Performance',
        4: 'Outstandng Performance'
    }

    result = performance_dict[prediction]

    return render_template(
        'index.html',
        prediction_text=f'Predicted Employee Performance: {result}'
    )

# RUN APP
if __name__ == '__main__':
    app.run(debug=True)