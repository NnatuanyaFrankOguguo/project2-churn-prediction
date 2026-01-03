"""
Interpretability Module for Customer Churn Prediction System
=============================================================
PHASE 3: THE "WHY" (INTERPRETABILITY)

This module handles:
1. SHAP TreeExplainer for XGBoost and Random Forest
2. Coefficient Analysis for Logistic Regression (fallback)
3. Local explanations for individual predictions
4. Global feature importance visualization
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Terminal-style color codes for logging
class LogColors:
    """ANSI color codes for terminal-style logging"""
    BLUE = '\033[94m'      # INFO
    GREEN = '\033[92m'     # SUCCESS
    YELLOW = '\033[93m'    # WARNING
    RED = '\033[91m'       # ERROR
    RESET = '\033[0m'      # Reset to default
    BOLD = '\033[1m'

def log_info(message: str):
    """Log informational messages in blue"""
    print(f"{LogColors.BLUE}[INFO]{LogColors.RESET} {message}")

def log_success(message: str):
    """Log success messages in green"""
    print(f"{LogColors.GREEN}[SUCCESS]{LogColors.RESET} {message}")

def log_warning(message: str):
    """Log warning messages in yellow"""
    print(f"{LogColors.YELLOW}[WARNING]{LogColors.RESET} {message}")

def log_error(message: str, root_cause: str = "", location: str = ""):
    """Log error messages in red with root cause analysis"""
    print(f"{LogColors.RED}[ERROR]{LogColors.RESET} {message}")
    if root_cause:
        print(f"{LogColors.RED}  └─ Root Cause:{LogColors.RESET} {root_cause}")
    if location:
        print(f"{LogColors.RED}  └─ Location:{LogColors.RESET} {location}")


class ChurnInterpretability:
    """
    Provides interpretability for churn prediction models.
    
    Supports:
    - SHAP TreeExplainer for tree-based models (XGBoost, Random Forest)
    - Coefficient analysis for Logistic Regression
    - Local (single-customer) and global (all-data) explanations
    """
    
    def __init__(self, models_dir: str, data_dir: str):
        """
        Initialize the interpretability module.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing processed data
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Loaded artifacts
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.X_train = None
        
        # SHAP explainers
        self.explainers = {}
        
        log_info(f"ChurnInterpretability initialized")
        log_info(f"  └─ Models directory: {models_dir}")
        log_info(f"  └─ Data directory: {data_dir}")
    
    def load_artifacts(self) -> None:
        """
        STEP 3.1: Load trained models and preprocessing artifacts
        """
        log_info("=" * 70)
        log_info("STEP 3.1: Loading Models and Artifacts")
        log_info("=" * 70)
        
        try:
            # Load models
            self.models['logistic'] = joblib.load(os.path.join(self.models_dir, "logistic.pkl"))
            self.models['random_forest'] = joblib.load(os.path.join(self.models_dir, "random_forest.pkl"))
            self.models['xgboost'] = joblib.load(os.path.join(self.models_dir, "xgboost.pkl"))
            log_success(f"Loaded 3 models: {list(self.models.keys())}")
            
            # Load scaler
            self.scaler = joblib.load(os.path.join(self.models_dir, "scaler.pkl"))
            log_success("Loaded scaler")
            
            # Load feature names
            self.feature_names = joblib.load(os.path.join(self.models_dir, "feature_names.pkl"))
            log_success(f"Loaded {len(self.feature_names)} feature names")
            
            # Load training data for SHAP background
            train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
            exclude_cols = ['Churn', 'CLV_quartile']
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]
            self.X_train = train_df[feature_cols].values
            log_success(f"Loaded training data: {self.X_train.shape}")
            
        except Exception as e:
            log_error(
                "Failed to load artifacts",
                root_cause=str(e),
                location="ChurnInterpretability.load_artifacts()"
            )
            raise
    
    def create_shap_explainers(self, sample_size: int = 200) -> None:
        """
        STEP 3.2: Create SHAP explainers for tree-based models
        
        Uses TreeExplainer for XGBoost and Random Forest.
        Logistic Regression uses coefficient analysis (more appropriate).
        
        Args:
            sample_size: Number of samples for background data (for speed)
        """
        log_info("=" * 70)
        log_info("STEP 3.2: Creating SHAP Explainers")
        log_info("=" * 70)
        
        log_warning(f"Using {sample_size} samples for SHAP background (for speed)")
        
        # Sample background data
        np.random.seed(42)
        indices = np.random.choice(len(self.X_train), size=min(sample_size, len(self.X_train)), replace=False)
        X_background = self.X_train[indices]
        
        # Create TreeExplainer for XGBoost
        log_info("Creating SHAP TreeExplainer for XGBoost...")
        try:
            self.explainers['xgboost'] = shap.TreeExplainer(self.models['xgboost'])
            log_success("XGBoost TreeExplainer created")
        except Exception as e:
            log_warning(f"Failed to create XGBoost explainer: {e}")
            log_info("Will use feature_importances_ as fallback")
        
        # Create TreeExplainer for Random Forest
        log_info("Creating SHAP TreeExplainer for Random Forest...")
        try:
            self.explainers['random_forest'] = shap.TreeExplainer(self.models['random_forest'])
            log_success("Random Forest TreeExplainer created")
        except Exception as e:
            log_warning(f"Failed to create Random Forest explainer: {e}")
            log_info("Will use feature_importances_ as fallback")
        
        # Note: Skip KernelExplainer for Logistic Regression - coefficients are more interpretable
        log_info("Logistic Regression: Using coefficient analysis (faster, more interpretable)")
        log_success("Explainer creation complete")
    
    def get_logistic_feature_importance(self) -> pd.DataFrame:
        """
        STEP 3.3: Get feature importance for Logistic Regression
        
        Uses standardized coefficients: importance = |coefficient * std_dev|
        This accounts for feature scaling and provides interpretable importance.
        
        Returns:
            DataFrame with feature names and their importance scores
        """
        log_info("=" * 70)
        log_info("STEP 3.3: Logistic Regression Coefficient Analysis")
        log_info("=" * 70)
        
        log_info("Formula: importance = |coefficient × feature_std_dev|")
        
        # Get coefficients
        coefficients = self.models['logistic'].coef_[0]
        
        # Get feature standard deviations from scaler
        feature_stds = self.scaler.scale_
        
        # Calculate standardized importance
        importance = np.abs(coefficients * feature_stds)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'std_dev': feature_stds,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        log_success("Coefficient analysis complete")
        log_info("Top 10 features:")
        for i, row in importance_df.head(10).iterrows():
            sign = "+" if row['coefficient'] > 0 else "-"
            log_info(f"  {importance_df.index.get_loc(i)+1}. {row['feature']}: {row['importance']:.4f} ({sign})")
        
        return importance_df
    
    def get_tree_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance for tree-based models
        
        Uses built-in feature_importances_ attribute.
        
        Args:
            model_name: 'random_forest' or 'xgboost'
            
        Returns:
            DataFrame with feature names and importance scores
        """
        log_info(f"Getting feature importance for {model_name}...")
        
        model = self.models[model_name]
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        log_success(f"{model_name} feature importance computed")
        
        return importance_df
    
    def explain_single_prediction_shap(self, X: np.ndarray, model_name: str = 'xgboost') -> Tuple[np.ndarray, float]:
        """
        STEP 3.4: Get SHAP explanation for a single prediction
        
        Args:
            X: Feature array for a single customer (shape: 1 x n_features)
            model_name: 'random_forest' or 'xgboost'
            
        Returns:
            Tuple of (shap_values, base_value)
        """
        if model_name not in self.explainers:
            log_warning(f"No SHAP explainer for {model_name}, using feature importance fallback")
            return None, None
        
        explainer = self.explainers[model_name]
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]  # Positive class
        
        return shap_values, base_value
    
    def explain_single_prediction_logistic(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """
        Get feature contribution for a single Logistic Regression prediction
        
        Uses: contribution = coefficient × scaled_feature_value
        
        Args:
            X_scaled: Scaled feature array for a single customer
            
        Returns:
            DataFrame with feature contributions
        """
        coefficients = self.models['logistic'].coef_[0]
        
        # Calculate contributions
        contributions = coefficients * X_scaled.flatten()
        
        contribution_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_scaled.flatten(),
            'coefficient': coefficients,
            'contribution': contributions,
            'abs_contribution': np.abs(contributions)
        }).sort_values('abs_contribution', ascending=False)
        
        return contribution_df
    
    def create_local_explanation_plot(self, X: np.ndarray, model_name: str = 'xgboost', 
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a waterfall or bar plot explaining a single prediction
        
        Args:
            X: Feature array for a single customer
            model_name: Model to explain
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        log_info(f"Creating local explanation plot for {model_name}...")
        
        if model_name == 'logistic':
            # Use coefficient-based explanation
            X_scaled = self.scaler.transform(X)
            contrib_df = self.explain_single_prediction_logistic(X_scaled)
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            top_features = contrib_df.head(10)
            colors = ['red' if c > 0 else 'blue' for c in top_features['contribution']]
            
            ax.barh(range(len(top_features)), top_features['contribution'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Contribution to Churn Probability')
            ax.set_title('Feature Contributions (Logistic Regression)')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
        else:
            # Use SHAP values
            shap_values, base_value = self.explain_single_prediction_shap(X, model_name)
            
            if shap_values is None:
                # Fallback to feature importance
                importance_df = self.get_tree_feature_importance(model_name)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = importance_df.head(10)
                ax.barh(range(len(top_features)), top_features['importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.invert_yaxis()
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'Feature Importance ({model_name})')
            else:
                # Create SHAP waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get top features by absolute SHAP value
                shap_flat = shap_values.flatten()
                feature_shap = list(zip(self.feature_names, shap_flat))
                feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
                
                top_features = feature_shap[:10]
                features = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                ax.barh(range(len(features)), values, color=colors)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel('SHAP Value (impact on churn prediction)')
                ax.set_title(f'SHAP Explanation ({model_name})')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            log_success(f"Saved explanation plot to {save_path}")
        
        return fig
    
    def create_global_importance_plot(self, model_name: str = 'xgboost',
                                       sample_size: int = 100,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        STEP 3.5: Create global feature importance plot
        
        For tree models: Uses SHAP summary plot
        For logistic: Uses standardized coefficients
        
        Args:
            model_name: Model to explain
            sample_size: Number of samples for SHAP computation
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        log_info(f"Creating global importance plot for {model_name}...")
        
        if model_name == 'logistic':
            importance_df = self.get_logistic_feature_importance()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = importance_df.head(15)
            
            colors = ['red' if c > 0 else 'blue' for c in top_features['coefficient']]
            ax.barh(range(len(top_features)), top_features['importance'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Standardized Importance |coef × std|')
            ax.set_title('Logistic Regression Feature Importance\n(Red = increases churn, Blue = decreases churn)')
            
        else:
            if model_name in self.explainers:
                # Use SHAP summary
                np.random.seed(42)
                indices = np.random.choice(len(self.X_train), size=min(sample_size, len(self.X_train)), replace=False)
                X_sample = self.X_train[indices]
                
                log_info(f"Computing SHAP values for {sample_size} samples...")
                shap_values = self.explainers[model_name].shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Create summary plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Calculate mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Ensure mean_abs_shap is 1D and convert to float
                if len(mean_abs_shap.shape) > 1:
                    mean_abs_shap = mean_abs_shap.flatten()
                mean_abs_shap = [float(x) for x in mean_abs_shap]
                
                feature_importance = list(zip(self.feature_names, mean_abs_shap))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                top_features = feature_importance[:15]
                features = [f[0] for f in top_features]
                values = [f[1] for f in top_features]
                
                ax.barh(range(len(features)), values, color='steelblue')
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.invert_yaxis()
                ax.set_xlabel('Mean |SHAP Value|')
                ax.set_title(f'{model_name.upper()} - Global Feature Importance (SHAP)')
                
                log_success("SHAP summary computed")
            else:
                # Fallback to built-in importance
                importance_df = self.get_tree_feature_importance(model_name)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                top_features = importance_df.head(15)
                
                ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.invert_yaxis()
                ax.set_xlabel('Feature Importance (Gini/Gain)')
                ax.set_title(f'{model_name.upper()} - Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            log_success(f"Saved global importance plot to {save_path}")
        
        return fig
    
    def get_prediction_explanation(self, customer_data: Dict, model_name: str = 'xgboost') -> Dict:
        """
        Get a complete explanation for a single customer prediction
        
        This is the main function used by the Streamlit app.
        
        Args:
            customer_data: Dictionary of customer features
            model_name: Model to use for prediction
            
        Returns:
            Dictionary containing:
            - prediction: churn probability
            - risk_level: Low/Medium/High
            - top_factors: List of (feature, contribution, direction) tuples
            - clv: Customer Lifetime Value
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        df = df[self.feature_names]
        X = df.values
        
        # Get prediction
        model = self.models[model_name]
        if model_name == 'logistic':
            X_scaled = self.scaler.transform(X)
            proba = model.predict_proba(X_scaled)[0, 1]
        else:
            proba = model.predict_proba(X)[0, 1]
        
        # Determine risk level
        if proba < 0.3:
            risk_level = "Low"
        elif proba < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Get feature explanations
        if model_name == 'logistic':
            X_scaled = self.scaler.transform(X)
            contrib_df = self.explain_single_prediction_logistic(X_scaled)
            top_factors = [
                (row['feature'], abs(row['contribution']), 'increases' if row['contribution'] > 0 else 'decreases')
                for _, row in contrib_df.head(5).iterrows()
            ]
        else:
            shap_values, _ = self.explain_single_prediction_shap(X, model_name)
            if shap_values is not None:
                shap_flat = shap_values.flatten()
                feature_shap = list(zip(self.feature_names, shap_flat))
                feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
                top_factors = [
                    (f[0], abs(f[1]), 'increases' if f[1] > 0 else 'decreases')
                    for f in feature_shap[:5]
                ]
            else:
                # Fallback
                importance_df = self.get_tree_feature_importance(model_name)
                top_factors = [
                    (row['feature'], row['importance'], 'affects')
                    for _, row in importance_df.head(5).iterrows()
                ]
        
        # Get CLV from customer data
        clv = customer_data.get('CLV', customer_data.get('MonthlyCharges', 0) * 12)
        
        return {
            'prediction': proba,
            'risk_level': risk_level,
            'top_factors': top_factors,
            'clv': clv
        }
    
    def run_phase3(self) -> None:
        """
        Execute the complete Phase 3 pipeline
        """
        log_info(f"\n{LogColors.BOLD}{'='*70}{LogColors.RESET}")
        log_info(f"{LogColors.BOLD}PHASE 3: THE 'WHY' (INTERPRETABILITY){LogColors.RESET}")
        log_info(f"{LogColors.BOLD}{'='*70}{LogColors.RESET}\n")
        
        try:
            # Execute all steps
            self.load_artifacts()
            self.create_shap_explainers()
            
            # Test logistic regression importance
            log_importance = self.get_logistic_feature_importance()
            
            # Test tree importance
            rf_importance = self.get_tree_feature_importance('random_forest')
            xgb_importance = self.get_tree_feature_importance('xgboost')
            
            # Create sample plots
            log_info("=" * 70)
            log_info("STEP 3.5: Creating Sample Explanation Plots")
            log_info("=" * 70)
            
            # Use first training sample for demo
            X_sample = self.X_train[0:1]
            
            # Create plots directory
            plots_dir = os.path.join(self.models_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Global importance plots
            for model_name in ['logistic', 'random_forest', 'xgboost']:
                self.create_global_importance_plot(
                    model_name,
                    save_path=os.path.join(plots_dir, f"global_{model_name}.png")
                )
            
            # Local explanation plot (sample)
            self.create_local_explanation_plot(
                X_sample,
                model_name='xgboost',
                save_path=os.path.join(plots_dir, "local_xgboost_sample.png")
            )
            
            # Final summary
            log_info("\n" + "=" * 70)
            log_success("PHASE 3 COMPLETE: Interpretability")
            log_info("=" * 70)
            log_info("Summary:")
            log_info(f"  └─ SHAP TreeExplainer created for: XGBoost, Random Forest")
            log_info(f"  └─ Coefficient analysis ready for: Logistic Regression")
            log_info(f"  └─ Sample plots saved to: {plots_dir}")
            log_success("Ready for Phase 4: Streamlit App")
            log_info("=" * 70 + "\n")
            
        except Exception as e:
            log_error(
                "PHASE 3 FAILED",
                root_cause=str(e),
                location="ChurnInterpretability.run_phase3()"
            )
            raise


# Utility functions for the Streamlit app
def load_interpretability_module(models_dir: str, data_dir: str) -> ChurnInterpretability:
    """
    Load and initialize the interpretability module for the Streamlit app.
    
    This is a helper function that creates the module, loads artifacts,
    and creates SHAP explainers.
    
    Args:
        models_dir: Directory containing trained models
        data_dir: Directory containing processed data
        
    Returns:
        Initialized ChurnInterpretability object
    """
    interp = ChurnInterpretability(models_dir, data_dir)
    interp.load_artifacts()
    interp.create_shap_explainers(sample_size=100)  # Smaller for app speed
    return interp


if __name__ == "__main__":
    """
    Main execution block for Phase 3
    """
    # Define paths
    MODELS_DIR = r"c:\project2-churn-prediction\models"
    DATA_DIR = r"c:\project2-churn-prediction\data\processed"
    
    # Initialize and run Phase 3
    interp = ChurnInterpretability(MODELS_DIR, DATA_DIR)
    interp.run_phase3()
    
    print("\n" + LogColors.GREEN + LogColors.BOLD + "✓ Phase 3 execution completed successfully!" + LogColors.RESET)
    print(LogColors.BLUE + "→ Ready to proceed to Phase 4: Streamlit App" + LogColors.RESET)
