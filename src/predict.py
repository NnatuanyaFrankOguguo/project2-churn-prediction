"""
Prediction Module for Customer Churn Prediction System
=======================================================
Utility functions for making predictions on new customers.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, Any

# Encoding mappings (must match Phase 1 LabelEncoder - alphabetically sorted)
ENCODING_MAPS = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'No phone service': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    },
    'tenure_bucket': {'0-6m': 0, '12-24m': 1, '24m+': 2, '6-12m': 3}
}


class ChurnPredictor:
    """
    A wrapper class for making churn predictions on new customers.
    
    Usage:
        predictor = ChurnPredictor('models/')
        result = predictor.predict(customer_data)
    """
    
    def __init__(self, models_dir: str):
        """
        Initialize the predictor by loading models and artifacts.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self._load_models()
    
    def _load_models(self):
        """Load all models and preprocessing artifacts"""
        self.models = {
            'logistic': joblib.load(os.path.join(self.models_dir, "logistic.pkl")),
            'random_forest': joblib.load(os.path.join(self.models_dir, "random_forest.pkl")),
            'xgboost': joblib.load(os.path.join(self.models_dir, "xgboost.pkl"))
        }
        self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(self.models_dir, "feature_names.pkl"))
    
    def _categorize_tenure(self, tenure: int) -> str:
        """Convert tenure to bucket"""
        if tenure <= 6:
            return '0-6m'
        elif tenure <= 12:
            return '6-12m'
        elif tenure <= 24:
            return '12-24m'
        else:
            return '24m+'
    
    def _create_features(self, customer: Dict) -> np.ndarray:
        """
        Convert customer dictionary to feature array.
        
        Args:
            customer: Dictionary with customer attributes
            
        Returns:
            Feature array ready for prediction
        """
        encoded = {}
        
        # Basic info
        encoded['gender'] = ENCODING_MAPS['gender'][customer['gender']]
        encoded['SeniorCitizen'] = customer['SeniorCitizen']
        encoded['Partner'] = ENCODING_MAPS['Partner'][customer['Partner']]
        encoded['Dependents'] = ENCODING_MAPS['Dependents'][customer['Dependents']]
        encoded['tenure'] = customer['tenure']
        
        # Phone services
        encoded['PhoneService'] = ENCODING_MAPS['PhoneService'][customer['PhoneService']]
        encoded['MultipleLines'] = ENCODING_MAPS['MultipleLines'][customer['MultipleLines']]
        
        # Internet services
        encoded['InternetService'] = ENCODING_MAPS['InternetService'][customer['InternetService']]
        encoded['OnlineSecurity'] = ENCODING_MAPS['OnlineSecurity'][customer['OnlineSecurity']]
        encoded['OnlineBackup'] = ENCODING_MAPS['OnlineBackup'][customer['OnlineBackup']]
        encoded['DeviceProtection'] = ENCODING_MAPS['DeviceProtection'][customer['DeviceProtection']]
        encoded['TechSupport'] = ENCODING_MAPS['TechSupport'][customer['TechSupport']]
        encoded['StreamingTV'] = ENCODING_MAPS['StreamingTV'][customer['StreamingTV']]
        encoded['StreamingMovies'] = ENCODING_MAPS['StreamingMovies'][customer['StreamingMovies']]
        
        # Billing
        encoded['Contract'] = ENCODING_MAPS['Contract'][customer['Contract']]
        encoded['PaperlessBilling'] = ENCODING_MAPS['PaperlessBilling'][customer['PaperlessBilling']]
        encoded['PaymentMethod'] = ENCODING_MAPS['PaymentMethod'][customer['PaymentMethod']]
        encoded['MonthlyCharges'] = customer['MonthlyCharges']
        encoded['TotalCharges'] = customer.get('TotalCharges', customer['tenure'] * customer['MonthlyCharges'])
        
        # Engineered features
        tenure_bucket = self._categorize_tenure(customer['tenure'])
        encoded['tenure_bucket'] = ENCODING_MAPS['tenure_bucket'][tenure_bucket]
        
        # Services count
        services = 0
        if customer['PhoneService'] == 'Yes':
            services += 1
        if customer['InternetService'] in ['DSL', 'Fiber optic']:
            services += 1
        for svc in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']:
            if customer[svc] == 'Yes':
                services += 1
        encoded['services_count'] = services
        
        # Monthly to total ratio
        expected_total = customer['tenure'] * customer['MonthlyCharges']
        encoded['monthly_to_total_ratio'] = encoded['TotalCharges'] / max(1, expected_total)
        
        # Risk flags
        encoded['flag_internet_no_tech_support'] = int(
            customer['InternetService'] in ['DSL', 'Fiber optic'] and 
            customer['TechSupport'] == 'No'
        )
        encoded['flag_fiber_high_charges'] = int(
            customer['InternetService'] == 'Fiber optic' and 
            customer['MonthlyCharges'] > 70
        )
        encoded['flag_short_tenure_monthly'] = int(
            customer['tenure'] <= 12 and 
            customer['Contract'] == 'Month-to-month'
        )
        
        # Expected tenure and CLV
        expected_tenure = 24 if customer['Contract'] in ['One year', 'Two year'] else 12
        encoded['expected_tenure'] = expected_tenure
        encoded['CLV'] = customer['MonthlyCharges'] * expected_tenure
        
        # Create array in correct order
        features = [encoded[f] for f in self.feature_names]
        return np.array([features])
    
    def predict(self, customer: Dict, model_name: str = 'ensemble') -> Dict:
        """
        Predict churn probability for a customer.
        
        Args:
            customer: Dictionary with customer attributes
            model_name: 'logistic', 'random_forest', 'xgboost', or 'ensemble'
            
        Returns:
            Dictionary with prediction, risk level, and CLV
        """
        features = self._create_features(customer)
        
        if model_name == 'ensemble':
            probas = []
            for name in ['logistic', 'random_forest', 'xgboost']:
                probas.append(self._predict_single(features, name))
            proba = np.mean(probas)
        else:
            proba = self._predict_single(features, model_name)
        
        # Determine risk level
        if proba < 0.3:
            risk_level = "Low"
        elif proba < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate CLV
        expected_tenure = 24 if customer['Contract'] in ['One year', 'Two year'] else 12
        clv = customer['MonthlyCharges'] * expected_tenure
        
        return {
            'churn_probability': proba,
            'risk_level': risk_level,
            'clv': clv,
            'expected_tenure': expected_tenure
        }
    
    def _predict_single(self, features: np.ndarray, model_name: str) -> float:
        """Get prediction from a single model"""
        model = self.models[model_name]
        
        if model_name == 'logistic':
            features_scaled = self.scaler.transform(features)
            proba = model.predict_proba(features_scaled)[0, 1]
        else:
            proba = model.predict_proba(features)[0, 1]
        
        return proba


# Example high-risk customer profile for testing
HIGH_RISK_PROFILE = {
    'gender': 'Male',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 105.0,
    'TotalCharges': 210.0
}

# Example low-risk customer profile for testing
LOW_RISK_PROFILE = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 48,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Bank transfer (automatic)',
    'MonthlyCharges': 85.0,
    'TotalCharges': 4080.0
}


if __name__ == "__main__":
    """Test the predictor with sample profiles"""
    MODELS_DIR = r"c:\project2-churn-prediction\models"
    
    predictor = ChurnPredictor(MODELS_DIR)
    
    print("=" * 70)
    print("CHURN PREDICTION TEST")
    print("=" * 70)
    
    # Test high-risk profile
    print("\n🔴 HIGH-RISK PROFILE:")
    print(f"  Senior citizen, month-to-month, fiber optic, no support, electronic check")
    result = predictor.predict(HIGH_RISK_PROFILE)
    print(f"  Churn Probability: {result['churn_probability']*100:.1f}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  CLV: ${result['clv']:,.2f}")
    
    # Test low-risk profile
    print("\n🟢 LOW-RISK PROFILE:")
    print(f"  Long tenure, two-year contract, all services, automatic payment")
    result = predictor.predict(LOW_RISK_PROFILE)
    print(f"  Churn Probability: {result['churn_probability']*100:.1f}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  CLV: ${result['clv']:,.2f}")
    
    # Individual model predictions for high-risk
    print("\n📊 INDIVIDUAL MODEL PREDICTIONS (High-Risk Profile):")
    for model in ['logistic', 'random_forest', 'xgboost']:
        result = predictor.predict(HIGH_RISK_PROFILE, model_name=model)
        print(f"  {model}: {result['churn_probability']*100:.1f}%")
    
    print("\n✅ Prediction tests complete!")
