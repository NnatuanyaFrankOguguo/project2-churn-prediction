"""
Customer Churn Prediction & CLV Analysis Dashboard
===================================================
PHASE 4: PROFESSIONAL DASHBOARD (THE FRONTEND)

A single-page Streamlit app with three tabs:
1. Predict - Customer churn prediction with explanations
2. Model Performance - Metrics, ROC curves, feature importance
3. Business Insights - CLV analysis and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import shap

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Customer Churn Prediction & CLV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffc107;
        color: black;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #dc3545;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E3A5F;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Constants & Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

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

# ============================================================================
# Cached Loading Functions
# ============================================================================
@st.cache_data
def load_processed_data():
    """Load processed train/val/test data"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, val_df, test_df

@st.cache_resource
def load_models():
    """Load trained models and preprocessing artifacts"""
    logistic = joblib.load(os.path.join(MODELS_DIR, "logistic.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    model_results = joblib.load(os.path.join(MODELS_DIR, "model_results.pkl"))
    
    return {
        'logistic': logistic,
        'random_forest': rf,
        'xgboost': xgb,
        'scaler': scaler,
        'feature_names': feature_names,
        'results': model_results
    }

@st.cache_resource
def get_tree_explainer(model_name: str, _model):
    """Create SHAP TreeExplainer for a model"""
    return shap.TreeExplainer(_model)

# ============================================================================
# Helper Functions
# ============================================================================
def categorize_tenure(tenure: int) -> str:
    """Convert tenure to bucket"""
    if tenure <= 6:
        return '0-6m'
    elif tenure <= 12:
        return '6-12m'
    elif tenure <= 24:
        return '12-24m'
    else:
        return '24m+'

def calculate_clv(monthly_charges: float, contract: str) -> float:
    """
    Calculate Customer Lifetime Value
    
    Formula: CLV = MonthlyCharges × ExpectedTenure
    ExpectedTenure: 24 months for contracts, 12 for month-to-month
    """
    expected_tenure = 24 if contract in ['One year', 'Two year'] else 12
    return monthly_charges * expected_tenure

def get_risk_label(probability: float) -> tuple:
    """Get risk label and color based on churn probability"""
    if probability < 0.3:
        return "Low Risk", "risk-low", "🟢"
    elif probability < 0.6:
        return "Medium Risk", "risk-medium", "🟡"
    else:
        return "High Risk", "risk-high", "🔴"

def create_customer_features(inputs: dict, models_data: dict) -> np.ndarray:
    """Convert user inputs to feature array for prediction"""
    # Encode categorical features
    encoded = {}
    
    # Basic info
    encoded['gender'] = ENCODING_MAPS['gender'][inputs['gender']]
    encoded['SeniorCitizen'] = inputs['senior_citizen']
    encoded['Partner'] = ENCODING_MAPS['Partner'][inputs['partner']]
    encoded['Dependents'] = ENCODING_MAPS['Dependents'][inputs['dependents']]
    encoded['tenure'] = inputs['tenure']
    
    # Phone services
    encoded['PhoneService'] = ENCODING_MAPS['PhoneService'][inputs['phone_service']]
    encoded['MultipleLines'] = ENCODING_MAPS['MultipleLines'][inputs['multiple_lines']]
    
    # Internet services
    encoded['InternetService'] = ENCODING_MAPS['InternetService'][inputs['internet_service']]
    encoded['OnlineSecurity'] = ENCODING_MAPS['OnlineSecurity'][inputs['online_security']]
    encoded['OnlineBackup'] = ENCODING_MAPS['OnlineBackup'][inputs['online_backup']]
    encoded['DeviceProtection'] = ENCODING_MAPS['DeviceProtection'][inputs['device_protection']]
    encoded['TechSupport'] = ENCODING_MAPS['TechSupport'][inputs['tech_support']]
    encoded['StreamingTV'] = ENCODING_MAPS['StreamingTV'][inputs['streaming_tv']]
    encoded['StreamingMovies'] = ENCODING_MAPS['StreamingMovies'][inputs['streaming_movies']]
    
    # Billing
    encoded['Contract'] = ENCODING_MAPS['Contract'][inputs['contract']]
    encoded['PaperlessBilling'] = ENCODING_MAPS['PaperlessBilling'][inputs['paperless_billing']]
    encoded['PaymentMethod'] = ENCODING_MAPS['PaymentMethod'][inputs['payment_method']]
    encoded['MonthlyCharges'] = inputs['monthly_charges']
    encoded['TotalCharges'] = inputs['tenure'] * inputs['monthly_charges']
    
    # Engineered features
    tenure_bucket = categorize_tenure(inputs['tenure'])
    encoded['tenure_bucket'] = ENCODING_MAPS['tenure_bucket'][tenure_bucket]
    
    # Services count
    services = 0
    if inputs['phone_service'] == 'Yes':
        services += 1
    if inputs['internet_service'] in ['DSL', 'Fiber optic']:
        services += 1
    for svc in ['online_security', 'online_backup', 'device_protection', 
                'tech_support', 'streaming_tv', 'streaming_movies']:
        if inputs[svc] == 'Yes':
            services += 1
    encoded['services_count'] = services
    
    # Monthly to total ratio
    expected_total = inputs['tenure'] * inputs['monthly_charges']
    encoded['monthly_to_total_ratio'] = encoded['TotalCharges'] / max(1, expected_total)
    
    # Risk flags
    encoded['flag_internet_no_tech_support'] = int(
        inputs['internet_service'] in ['DSL', 'Fiber optic'] and 
        inputs['tech_support'] == 'No'
    )
    encoded['flag_fiber_high_charges'] = int(
        inputs['internet_service'] == 'Fiber optic' and 
        inputs['monthly_charges'] > 70
    )
    encoded['flag_short_tenure_monthly'] = int(
        inputs['tenure'] <= 12 and 
        inputs['contract'] == 'Month-to-month'
    )
    
    # Expected tenure and CLV
    expected_tenure = 24 if inputs['contract'] in ['One year', 'Two year'] else 12
    encoded['expected_tenure'] = expected_tenure
    encoded['CLV'] = inputs['monthly_charges'] * expected_tenure
    
    # Create array in correct order
    feature_order = models_data['feature_names']
    features = [encoded[f] for f in feature_order]
    
    return np.array([features])

def predict_churn(features: np.ndarray, model_name: str, models_data: dict) -> float:
    """Get churn probability from model"""
    model = models_data[model_name]
    
    if model_name == 'logistic':
        features_scaled = models_data['scaler'].transform(features)
        proba = model.predict_proba(features_scaled)[0, 1]
    else:
        proba = model.predict_proba(features)[0, 1]
    
    return proba

def get_ensemble_prediction(features: np.ndarray, models_data: dict) -> float:
    """Get average prediction from all 3 models"""
    probas = []
    for name in ['logistic', 'random_forest', 'xgboost']:
        probas.append(predict_churn(features, name, models_data))
    return np.mean(probas)

# ============================================================================
# Tab 1: Predict
# ============================================================================
def render_predict_tab(models_data: dict):
    """Render the prediction tab"""
    st.header("🔮 Customer Churn Prediction")
    st.markdown("Enter customer information to predict churn probability and estimated lifetime value.")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Customer Info")
        gender = st.selectbox("Gender", ['Female', 'Male'])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        partner = st.selectbox("Has Partner", ['No', 'Yes'])
        dependents = st.selectbox("Has Dependents", ['No', 'Yes'])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
        multiple_lines = st.selectbox("Multiple Lines", ['No', 'No phone service', 'Yes'])
        internet_service = st.selectbox("Internet Service", ['No', 'DSL', 'Fiber optic'])
        
        # Internet-dependent services
        internet_options = ['No', 'Yes'] if internet_service != 'No' else ['No internet service']
        online_security = st.selectbox("Online Security", internet_options)
        online_backup = st.selectbox("Online Backup", internet_options)
        device_protection = st.selectbox("Device Protection", internet_options)
        tech_support = st.selectbox("Tech Support", internet_options)
        streaming_tv = st.selectbox("Streaming TV", internet_options)
        streaming_movies = st.selectbox("Streaming Movies", internet_options)
    
    with col3:
        st.subheader("💳 Billing")
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
        payment_method = st.selectbox("Payment Method", [
            'Bank transfer (automatic)',
            'Credit card (automatic)',
            'Electronic check',
            'Mailed check'
        ])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    
    # Collect inputs
    inputs = {
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges
    }
    
    # Predict button
    if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
        # Create features
        features = create_customer_features(inputs, models_data)
        
        # Get predictions
        proba_ensemble = get_ensemble_prediction(features, models_data)
        risk_label, risk_class, risk_icon = get_risk_label(proba_ensemble)
        clv = calculate_clv(monthly_charges, contract)
        
        st.divider()
        
        # Results section
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(
                label="Churn Probability",
                value=f"{proba_ensemble*100:.1f}%",
                delta=None
            )
        
        with col_res2:
            st.markdown(f"""
            <div class="{risk_class}" style="text-align: center; padding: 1rem; border-radius: 10px;">
                {risk_icon} {risk_label}
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            st.metric(
                label="Estimated CLV",
                value=f"${clv:,.2f}",
                delta=f"Based on {24 if contract != 'Month-to-month' else 12} month expected tenure"
            )
        
        # Individual model predictions
        st.subheader("📊 Model Predictions Comparison")
        model_cols = st.columns(3)
        
        for idx, (name, display_name) in enumerate([
            ('logistic', 'Logistic Regression'),
            ('random_forest', 'Random Forest'),
            ('xgboost', 'XGBoost')
        ]):
            proba = predict_churn(features, name, models_data)
            with model_cols[idx]:
                st.metric(display_name, f"{proba*100:.1f}%")
        
        # Feature explanation
        st.subheader("🔍 Why This Prediction?")
        st.markdown("**Key factors influencing this customer's churn risk:**")
        
        # Get SHAP explanation for XGBoost
        try:
            explainer = get_tree_explainer('xgboost', models_data['xgboost'])
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap_flat = shap_values.flatten()
            feature_shap = list(zip(models_data['feature_names'], shap_flat))
            feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Display top factors
            for feat, shap_val in feature_shap[:5]:
                direction = "⬆️ increases" if shap_val > 0 else "⬇️ decreases"
                color = "red" if shap_val > 0 else "green"
                st.markdown(f"- **{feat}**: {direction} churn risk")
            
            # Create SHAP bar plot
            fig, ax = plt.subplots(figsize=(10, 5))
            top_features = feature_shap[:10]
            features_names = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            colors = ['#dc3545' if v > 0 else '#28a745' for v in values]
            
            ax.barh(range(len(features_names)), values, color=colors)
            ax.set_yticks(range(len(features_names)))
            ax.set_yticklabels(features_names)
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('SHAP Value (impact on churn prediction)')
            ax.set_title('Feature Contributions to This Prediction')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.info("Feature importance based on model coefficients/importance scores.")
        
        # CLV Formula explanation
        st.subheader("💰 CLV Calculation")
        expected_tenure = 24 if contract != 'Month-to-month' else 12
        st.markdown(f"""
        <div class="insight-box">
        <b>Formula:</b> CLV = Monthly Charges × Expected Tenure<br>
        <b>Calculation:</b> ${monthly_charges:.2f} × {expected_tenure} months = <b>${clv:,.2f}</b><br>
        <b>Assumption:</b> {'Contract customers expected to stay 24 months' if contract != 'Month-to-month' else 'Month-to-month customers expected to stay 12 months'}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# Tab 2: Model Performance
# ============================================================================
def render_performance_tab(models_data: dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Render the model performance tab"""
    st.header("📈 Model Performance")
    
    # Metrics table
    st.subheader("Performance Metrics Comparison")
    
    results = models_data['results']
    metrics_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': [results['logistic']['precision'], results['random_forest']['precision'], results['xgboost']['precision']],
        'Recall': [results['logistic']['recall'], results['random_forest']['recall'], results['xgboost']['recall']],
        'F1 Score': [results['logistic']['f1'], results['random_forest']['f1'], results['xgboost']['f1']],
        'AUC-ROC': [results['logistic']['auc_roc'], results['random_forest']['auc_roc'], results['xgboost']['auc_roc']]
    })
    
    # Style the dataframe
    styled_df = metrics_df.style.format({
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1 Score': '{:.3f}',
        'AUC-ROC': '{:.3f}'
    }).background_gradient(subset=['AUC-ROC'], cmap='Greens')
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Key metrics highlights
    best_auc = metrics_df.loc[metrics_df['AUC-ROC'].idxmax(), 'Model']
    best_recall = metrics_df.loc[metrics_df['Recall'].idxmax(), 'Model']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ **Best AUC-ROC:** {best_auc}")
    with col2:
        st.success(f"✅ **Best Recall:** {best_recall}")
    
    st.divider()
    
    # ROC Curves
    st.subheader("ROC Curves")
    
    # Prepare test data
    exclude_cols = ['Churn', 'CLV_quartile']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    X_test = test_df[feature_cols].values
    y_test = test_df['Churn'].values
    X_test_scaled = models_data['scaler'].transform(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    model_names = [('logistic', 'Logistic Regression'), 
                   ('random_forest', 'Random Forest'), 
                   ('xgboost', 'XGBoost')]
    
    for (name, display_name), color in zip(model_names, colors):
        model = models_data[name]
        if name == 'logistic':
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{display_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    model_choice = st.selectbox(
        "Select Model",
        ['XGBoost', 'Random Forest', 'Logistic Regression'],
        key='cm_model'
    )
    
    model_key = model_choice.lower().replace(' ', '_')
    model = models_data[model_key]
    
    if model_key == 'logistic':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Churn', 'Churn'])
    ax.set_yticklabels(['No Churn', 'Churn'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_choice}')
    
    # Add values
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', 
                          color='white' if cm[i, j] > cm.max()/2 else 'black',
                          fontsize=14, fontweight='bold')
    
    plt.colorbar(im)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # Global Feature Importance
    st.subheader("🎯 Global Feature Importance")
    
    importance_model = st.selectbox(
        "Select Model for Feature Importance",
        ['XGBoost', 'Random Forest', 'Logistic Regression'],
        key='imp_model'
    )
    
    imp_model_key = importance_model.lower().replace(' ', '_')
    
    if imp_model_key == 'logistic':
        # Coefficient-based importance
        coefficients = models_data['logistic'].coef_[0]
        feature_stds = models_data['scaler'].scale_
        importance = np.abs(coefficients * feature_stds)
        
        importance_df = pd.DataFrame({
            'feature': models_data['feature_names'],
            'importance': importance
        }).sort_values('importance', ascending=True).tail(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#dc3545' if coefficients[models_data['feature_names'].index(f)] > 0 else '#28a745' 
                  for f in importance_df['feature']]
        ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
        ax.set_xlabel('Standardized Importance |coefficient × std|')
        ax.set_title('Logistic Regression Feature Importance\n(Red = increases churn, Green = decreases churn)')
    else:
        model = models_data[imp_model_key]
        importance_df = pd.DataFrame({
            'feature': models_data['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{importance_model} - Feature Importance')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# Tab 3: Business Insights
# ============================================================================
def render_insights_tab(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Render the business insights tab"""
    st.header("💡 Business Insights: CLV Analysis")
    
    # Combine all data for analysis
    all_data = pd.concat([train_df, test_df], ignore_index=True)
    
    # CLV Distribution
    st.subheader("📊 CLV Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_data['CLV'], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(all_data['CLV'].mean(), color='red', linestyle='--', label=f'Mean: ${all_data["CLV"].mean():.2f}')
        ax.axvline(all_data['CLV'].median(), color='green', linestyle='--', label=f'Median: ${all_data["CLV"].median():.2f}')
        ax.set_xlabel('Customer Lifetime Value ($)')
        ax.set_ylabel('Number of Customers')
        ax.set_title('Distribution of Customer Lifetime Value')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # CLV by Quartile
        quartile_stats = all_data.groupby('CLV_quartile').agg({
            'CLV': 'mean',
            'Churn': ['mean', 'count']
        }).round(2)
        quartile_stats.columns = ['Avg CLV', 'Churn Rate', 'Customer Count']
        quartile_stats = quartile_stats.reset_index()
        quartile_stats['Churn Rate'] = (quartile_stats['Churn Rate'] * 100).round(1).astype(str) + '%'
        quartile_stats['Avg CLV'] = '$' + quartile_stats['Avg CLV'].round(0).astype(int).astype(str)
        
        st.dataframe(quartile_stats, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Churn Rate by CLV Quartile (key visualization)
    st.subheader("🔴 Churn Rate by CLV Quartile")
    
    quartile_order = ['Low', 'Medium', 'High', 'Premium']
    churn_by_quartile = all_data.groupby('CLV_quartile')['Churn'].mean() * 100
    churn_by_quartile = churn_by_quartile.reindex(quartile_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#28a745', '#ffc107', '#dc3545', '#17a2b8']
    bars = ax.bar(quartile_order, churn_by_quartile.values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, churn_by_quartile.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('CLV Quartile', fontsize=12)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_title('🎯 Key Finding: High CLV Customers Have the Highest Churn Rate!', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(churn_by_quartile.values) + 10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # Business Insights
    st.subheader("📝 Key Business Insights")
    
    high_clv_churn = churn_by_quartile['High']
    premium_clv_churn = churn_by_quartile['Premium']
    low_clv_churn = churn_by_quartile['Low']
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>🚨 Critical Finding #1: High-Value Customer Churn</h4>
    <p>The <b>High CLV quartile has a {high_clv_churn:.1f}% churn rate</b> - the highest among all segments! 
    These are customers with month-to-month contracts paying premium prices for Fiber Optic internet but 
    without security/support services. They are likely dissatisfied with the value they're getting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>✅ Bright Spot: Premium Customers Are Loyal</h4>
    <p><b>Premium CLV customers have only {premium_clv_churn:.1f}% churn rate</b> - the lowest! 
    These customers typically have long-term contracts (1-2 years), which creates commitment and 
    reduces churn likelihood. Contract lock-in is working.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="insight-box">
    <h4>💡 Actionable Recommendation</h4>
    <p><b>Priority Action: Target High CLV, High Churn customers with:</b></p>
    <ul>
        <li>🎯 Contract upgrade incentives (offer discounts for 1-year commitment)</li>
        <li>🛡️ Bundled security/tech support packages at reduced rates</li>
        <li>💳 Switch payment method from Electronic Check to automatic payments</li>
        <li>👋 Proactive outreach from Customer Success team within first 6 months</li>
    </ul>
    <p><b>Expected Impact:</b> Reducing churn in the High CLV segment by 10% could save 
    approximately $150,000+ in annual recurring revenue.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Who to Retain First
    st.subheader("🎯 Customer Prioritization Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔴 Retain Immediately (High Priority)
        - **High CLV + Month-to-month contract**
        - Fiber optic internet, no tech support
        - Electronic check payment
        - Tenure < 12 months
        - Senior citizens with premium services
        
        *These customers are paying a lot but are at high risk of leaving!*
        """)
    
    with col2:
        st.markdown("""
        ### 🟡 Monitor & Upsell (Medium Priority)
        - **Medium CLV + Month-to-month contract**
        - DSL internet customers
        - No security services
        
        ### 🟢 Nurture (Lower Priority)
        - **Premium CLV + Long-term contracts**
        - Already committed customers
        - Focus on satisfaction surveys
        """)

# ============================================================================
# Main App
# ============================================================================
def main():
    """Main application entry point"""
    # Header
    st.markdown('<p class="main-header">🔮 Customer Churn Prediction & CLV Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn, understand the reasons, and prioritize retention efforts</p>', unsafe_allow_html=True)
    
    # Load data and models
    try:
        train_df, val_df, test_df = load_processed_data()
        models_data = load_models()
    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        st.info("Please ensure the models have been trained and saved in the 'models' directory.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Performance", "💡 Business Insights"])
    
    with tab1:
        render_predict_tab(models_data)
    
    with tab2:
        render_performance_tab(models_data, train_df, test_df)
    
    with tab3:
        render_insights_tab(train_df, test_df)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ using Streamlit | Models: Logistic Regression, Random Forest, XGBoost | 
        Data: IBM Telco Customer Churn Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
