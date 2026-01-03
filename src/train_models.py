"""
Model Training Module for Customer Churn Prediction System
=============================================================
PHASE 2: THE MODEL TRIO (INTELLIGENCE)

This module handles:
1. Training Logistic Regression (Baseline)
2. Training Random Forest
3. Training XGBoost
4. Light hyperparameter tuning
5. Model evaluation with Precision, Recall, F1, AUC-ROC
6. Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, Any
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


class ChurnModelTrainer:
    """
    Handles all model training operations for the Churn Prediction system.
    
    Trains three models:
    1. Logistic Regression (baseline, interpretable)
    2. Random Forest (ensemble, feature importance)
    3. XGBoost (gradient boosting, high performance)
    """
    
    def __init__(self, data_dir: str, models_dir: str):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Directory containing processed train/val/test data
            models_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Data containers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Preprocessing
        self.scaler = None
        self.feature_names = None
        
        # Trained models
        self.models = {}
        self.results = {}
        
        log_info(f"ChurnModelTrainer initialized")
        log_info(f"  └─ Data directory: {data_dir}")
        log_info(f"  └─ Models directory: {models_dir}")
    
    def load_data(self) -> None:
        """
        STEP 2.1: Load processed train/val/test data
        
        Loads the CSV files saved by Phase 1 and prepares features/targets.
        """
        log_info("=" * 70)
        log_info("STEP 2.1: Loading Processed Data")
        log_info("=" * 70)
        
        try:
            # Load splits
            train_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
            val_df = pd.read_csv(os.path.join(self.data_dir, "val.csv"))
            test_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
            
            log_success(f"Loaded train.csv: {train_df.shape}")
            log_success(f"Loaded val.csv: {val_df.shape}")
            log_success(f"Loaded test.csv: {test_df.shape}")
            
            # Columns to exclude from features
            # CLV_quartile is categorical (for analysis), not for model training
            exclude_cols = ['Churn', 'CLV_quartile']
            
            # Extract features and targets
            self.feature_names = [col for col in train_df.columns if col not in exclude_cols]
            
            self.X_train = train_df[self.feature_names].values
            self.y_train = train_df['Churn'].values
            
            self.X_val = val_df[self.feature_names].values
            self.y_val = val_df['Churn'].values
            
            self.X_test = test_df[self.feature_names].values
            self.y_test = test_df['Churn'].values
            
            log_info(f"Feature count: {len(self.feature_names)}")
            log_info(f"Features: {self.feature_names}")
            
            # Check class distribution
            train_churn_rate = self.y_train.mean()
            log_info(f"Training set churn rate: {train_churn_rate*100:.2f}%")
            
            if train_churn_rate < 0.30:
                log_warning(f"Class imbalance detected! Will use class_weight='balanced'")
            
            log_success("Data loading complete")
            
        except Exception as e:
            log_error(
                "Failed to load data",
                root_cause=str(e),
                location="ChurnModelTrainer.load_data()"
            )
            raise
    
    def scale_features(self) -> None:
        """
        STEP 2.2: Scale features using StandardScaler
        
        Important: Fit scaler on training data only, then transform all sets.
        """
        log_info("=" * 70)
        log_info("STEP 2.2: Feature Scaling")
        log_info("=" * 70)
        
        # Initialize and fit scaler on training data only
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        log_info("Scaler fitted on training data")
        log_info(f"  └─ Feature means: {self.scaler.mean_[:5]}... (first 5)")
        log_info(f"  └─ Feature stds: {self.scaler.scale_[:5]}... (first 5)")
        
        # Transform validation and test sets
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        log_success("All datasets scaled")
        log_warning("Remember: scaler was fitted on training data only (no data leakage)")
    
    def train_logistic_regression(self) -> Dict[str, float]:
        """
        STEP 2.3: Train Logistic Regression (Baseline Model)
        
        A simple, interpretable model that serves as our baseline.
        Uses class_weight='balanced' to handle imbalance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        log_info("=" * 70)
        log_info("STEP 2.3: Training Logistic Regression (Baseline)")
        log_info("=" * 70)
        
        log_info("Hyperparameters:")
        log_info("  └─ C (regularization): Grid search over [0.01, 0.1, 1, 10]")
        log_info("  └─ class_weight: 'balanced' (handles imbalance)")
        log_info("  └─ max_iter: 1000")
        log_info("  └─ solver: 'lbfgs'")
        
        # Define model with class_weight to handle imbalance
        base_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        
        # Light hyperparameter tuning
        param_grid = {
            'C': [0.01, 0.1, 1, 10]
        }
        
        log_info("Running GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train on scaled data (logistic regression benefits from scaling)
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_model = grid_search.best_estimator_
        log_success(f"Best hyperparameters: {grid_search.best_params_}")
        log_info(f"  └─ Best CV AUC: {grid_search.best_score_:.4f}")
        
        # Store model
        self.models['logistic'] = best_model
        
        # Evaluate on test set
        metrics = self._evaluate_model(best_model, self.X_test_scaled, self.y_test, "Logistic Regression")
        self.results['logistic'] = metrics
        
        # Log feature importances (coefficients)
        log_info("Top 10 Most Important Features (by absolute coefficient):")
        coef_importance = np.abs(best_model.coef_[0]) * self.scaler.scale_
        feature_importance = list(zip(self.feature_names, coef_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feat, imp) in enumerate(feature_importance[:10]):
            log_info(f"  {i+1}. {feat}: {imp:.4f}")
        
        return metrics
    
    def train_random_forest(self) -> Dict[str, float]:
        """
        STEP 2.4: Train Random Forest
        
        An ensemble model that provides built-in feature importance.
        Uses class_weight='balanced' to handle imbalance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        log_info("=" * 70)
        log_info("STEP 2.4: Training Random Forest")
        log_info("=" * 70)
        
        log_info("Hyperparameters to tune:")
        log_info("  └─ max_depth: [5, 10, 15, None]")
        log_info("  └─ min_samples_leaf: [1, 2, 4]")
        log_info("  └─ n_estimators: 100 (fixed for speed)")
        log_info("  └─ class_weight: 'balanced'")
        
        # Define model
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Light hyperparameter tuning
        param_grid = {
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [1, 2, 4]
        }
        
        log_info("Running GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train on unscaled data (trees don't need scaling)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        log_success(f"Best hyperparameters: {grid_search.best_params_}")
        log_info(f"  └─ Best CV AUC: {grid_search.best_score_:.4f}")
        
        # Store model
        self.models['random_forest'] = best_model
        
        # Evaluate on test set
        metrics = self._evaluate_model(best_model, self.X_test, self.y_test, "Random Forest")
        self.results['random_forest'] = metrics
        
        # Log feature importances
        log_info("Top 10 Most Important Features (Gini Importance):")
        feature_importance = list(zip(self.feature_names, best_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feat, imp) in enumerate(feature_importance[:10]):
            log_info(f"  {i+1}. {feat}: {imp:.4f}")
        
        return metrics
    
    def train_xgboost(self) -> Dict[str, float]:
        """
        STEP 2.5: Train XGBoost
        
        A gradient boosting model known for high performance.
        Uses scale_pos_weight to handle class imbalance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        log_info("=" * 70)
        log_info("STEP 2.5: Training XGBoost")
        log_info("=" * 70)
        
        # Calculate scale_pos_weight for imbalance
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        log_info(f"Class imbalance: {neg_count} negative, {pos_count} positive")
        log_info(f"  └─ scale_pos_weight: {scale_pos_weight:.2f}")
        
        log_info("Hyperparameters to tune:")
        log_info("  └─ max_depth: [3, 5, 7]")
        log_info("  └─ learning_rate: [0.01, 0.1, 0.2]")
        log_info("  └─ n_estimators: 100 (fixed for speed)")
        
        # Define model
        base_model = xgb.XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Light hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        log_info("Running GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train on unscaled data (trees don't need scaling)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        log_success(f"Best hyperparameters: {grid_search.best_params_}")
        log_info(f"  └─ Best CV AUC: {grid_search.best_score_:.4f}")
        
        # Store model
        self.models['xgboost'] = best_model
        
        # Evaluate on test set
        metrics = self._evaluate_model(best_model, self.X_test, self.y_test, "XGBoost")
        self.results['xgboost'] = metrics
        
        # Log feature importances
        log_info("Top 10 Most Important Features (XGBoost Importance):")
        feature_importance = list(zip(self.feature_names, best_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feat, imp) in enumerate(feature_importance[:10]):
            log_info(f"  {i+1}. {feat}: {imp:.4f}")
        
        return metrics
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate a model on the given dataset.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            model_name: Name for logging
            
        Returns:
            Dictionary containing Precision, Recall, F1, AUC-ROC
        """
        log_info(f"\nEvaluating {model_name} on test set...")
        
        # Get predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc_roc = roc_auc_score(y, y_proba)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
        
        log_success(f"{model_name} Test Set Metrics:")
        log_info(f"  └─ Precision: {precision:.4f}")
        log_info(f"  └─ Recall: {recall:.4f}")
        log_info(f"  └─ F1 Score: {f1:.4f}")
        log_info(f"  └─ AUC-ROC: {auc_roc:.4f}")
        
        # Check against targets
        if recall < 0.60:
            log_warning(f"Recall ({recall:.4f}) is below target (0.60). Consider more tuning.")
        else:
            log_success(f"Recall meets target (>= 0.60)")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        log_info(f"Confusion Matrix:")
        log_info(f"  └─ TN: {cm[0,0]}, FP: {cm[0,1]}")
        log_info(f"  └─ FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return metrics
    
    def verify_high_risk_profile(self) -> None:
        """
        STEP 2.6: Verify High-Risk Profile Detection
        
        Test case: Senior citizen with month-to-month contract, fiber optic internet,
        no security/backup/tech support, paying by electronic check, monthly charges >= $100
        Expected: >60% churn probability
        """
        log_info("=" * 70)
        log_info("STEP 2.6: Verifying High-Risk Profile Detection")
        log_info("=" * 70)
        
        log_info("Creating test profile:")
        log_info("  └─ Senior Citizen: Yes (1)")
        log_info("  └─ Contract: Month-to-month (0)")
        log_info("  └─ Internet Service: Fiber optic (1)")
        log_info("  └─ Tech Support: No (0)")
        log_info("  └─ Online Security: No (0)")
        log_info("  └─ Online Backup: No (0)")
        log_info("  └─ Payment Method: Electronic check (2)")
        log_info("  └─ Monthly Charges: $100+")
        
        # Use the feature names from training (self.feature_names)
        # These are already loaded and exclude CLV_quartile
        
        # Set high-risk values
        # Based on Phase 1 encoding:
        # Contract: Month-to-month=0
        # InternetService: Fiber optic=1
        # TechSupport: No=0
        # PaymentMethod: Electronic check=2
        # OnlineSecurity: No=0
        # OnlineBackup: No=0
        
        test_data = {
            'gender': 1,  # Male (doesn't matter much)
            'SeniorCitizen': 1,  # Yes
            'Partner': 0,  # No
            'Dependents': 0,  # No
            'tenure': 2,  # Short tenure (high risk)
            'PhoneService': 1,  # Yes
            'MultipleLines': 0,  # No
            'InternetService': 1,  # Fiber optic (high risk)
            'OnlineSecurity': 0,  # No (high risk)
            'OnlineBackup': 0,  # No (high risk)
            'DeviceProtection': 0,  # No
            'TechSupport': 0,  # No (high risk)
            'StreamingTV': 0,  # No
            'StreamingMovies': 0,  # No
            'Contract': 0,  # Month-to-month (high risk)
            'PaperlessBilling': 1,  # Yes
            'PaymentMethod': 2,  # Electronic check (high risk)
            'MonthlyCharges': 105.0,  # >= $100 (high risk)
            'TotalCharges': 210.0,  # 2 months * 105
            'tenure_bucket': 0,  # 0-6m (high risk)
            'services_count': 2,  # Low
            'monthly_to_total_ratio': 1.0,
            'flag_internet_no_tech_support': 1,  # Flag set
            'flag_fiber_high_charges': 1,  # Flag set
            'flag_short_tenure_monthly': 1,  # Flag set
            'expected_tenure': 12,  # Month-to-month
            'CLV': 1260.0  # 105 * 12
        }
        
        # Create DataFrame with correct feature order (use self.feature_names)
        test_profile = pd.DataFrame([test_data])
        
        # Ensure columns are in the right order
        test_profile = test_profile[self.feature_names]
        
        log_info("\nRunning predictions on high-risk profile...")
        
        all_passed = True
        
        for name, model in self.models.items():
            # Use scaled data for logistic regression
            if name == 'logistic':
                X_test = self.scaler.transform(test_profile.values)
            else:
                X_test = test_profile.values
            
            proba = model.predict_proba(X_test)[0, 1]
            
            if proba >= 0.60:
                log_success(f"{name.upper()}: {proba*100:.1f}% churn probability (✓ PASS)")
            else:
                log_warning(f"{name.upper()}: {proba*100:.1f}% churn probability (✗ FAIL - expected >60%)")
                all_passed = False
        
        if all_passed:
            log_success("\n✓ All models correctly identify high-risk profile!")
        else:
            log_warning("\n⚠ Some models did not meet the 60% threshold for high-risk profile")
            log_info("  This may be acceptable if other metrics are strong")
    
    def compare_models(self) -> None:
        """
        STEP 2.7: Compare all models and print summary
        """
        log_info("=" * 70)
        log_info("STEP 2.7: Model Comparison Summary")
        log_info("=" * 70)
        
        # Create comparison table
        log_info("\n" + "=" * 70)
        log_info(f"{'Model':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC-ROC':>12}")
        log_info("=" * 70)
        
        best_auc = 0
        best_model_name = ""
        
        for name, metrics in self.results.items():
            log_info(f"{name:<20} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} {metrics['f1']:>12.4f} {metrics['auc_roc']:>12.4f}")
            
            if metrics['auc_roc'] > best_auc:
                best_auc = metrics['auc_roc']
                best_model_name = name
        
        log_info("=" * 70)
        log_success(f"Best model by AUC-ROC: {best_model_name} ({best_auc:.4f})")
        
        # Verify RF and XGB beat Logistic Regression
        if 'logistic' in self.results:
            lr_auc = self.results['logistic']['auc_roc']
            for name in ['random_forest', 'xgboost']:
                if name in self.results:
                    if self.results[name]['auc_roc'] > lr_auc:
                        log_success(f"{name} beats Logistic Regression: {self.results[name]['auc_roc']:.4f} > {lr_auc:.4f}")
                    else:
                        log_warning(f"{name} does not beat Logistic Regression: {self.results[name]['auc_roc']:.4f} <= {lr_auc:.4f}")
    
    def save_models(self) -> None:
        """
        STEP 2.8: Save trained models and preprocessing artifacts
        """
        log_info("=" * 70)
        log_info("STEP 2.8: Saving Models and Artifacts")
        log_info("=" * 70)
        
        # Create models directory if needed
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            log_success(f"Saved {name}.pkl")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        log_success("Saved scaler.pkl")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, features_path)
        log_success("Saved feature_names.pkl")
        
        # Save results
        results_path = os.path.join(self.models_dir, "model_results.pkl")
        joblib.dump(self.results, results_path)
        log_success("Saved model_results.pkl")
        
        log_success(f"All models saved to: {self.models_dir}")
    
    def run_phase2(self) -> None:
        """
        Execute the complete Phase 2 pipeline
        """
        log_info(f"\n{LogColors.BOLD}{'='*70}{LogColors.RESET}")
        log_info(f"{LogColors.BOLD}PHASE 2: THE MODEL TRIO (INTELLIGENCE){LogColors.RESET}")
        log_info(f"{LogColors.BOLD}{'='*70}{LogColors.RESET}\n")
        
        try:
            # Execute all steps
            self.load_data()
            self.scale_features()
            self.train_logistic_regression()
            self.train_random_forest()
            self.train_xgboost()
            self.verify_high_risk_profile()
            self.compare_models()
            self.save_models()
            
            # Final summary
            log_info("\n" + "=" * 70)
            log_success("PHASE 2 COMPLETE: The Model Trio")
            log_info("=" * 70)
            log_info("Summary:")
            log_info(f"  └─ Models trained: {list(self.models.keys())}")
            log_info(f"  └─ Best AUC-ROC: {max(m['auc_roc'] for m in self.results.values()):.4f}")
            log_info(f"  └─ Best Recall: {max(m['recall'] for m in self.results.values()):.4f}")
            log_success(f"All models saved to: {self.models_dir}")
            log_info("=" * 70 + "\n")
            
        except Exception as e:
            log_error(
                "PHASE 2 FAILED",
                root_cause=str(e),
                location="ChurnModelTrainer.run_phase2()"
            )
            raise


if __name__ == "__main__":
    """
    Main execution block for Phase 2
    """
    # Define paths
    DATA_DIR = r"c:\project2-churn-prediction\data\processed"
    MODELS_DIR = r"c:\project2-churn-prediction\models"
    
    # Initialize and run Phase 2
    trainer = ChurnModelTrainer(DATA_DIR, MODELS_DIR)
    trainer.run_phase2()
    
    print("\n" + LogColors.GREEN + LogColors.BOLD + "✓ Phase 2 execution completed successfully!" + LogColors.RESET)
    print(LogColors.BLUE + "→ Ready to proceed to Phase 3: Interpretability" + LogColors.RESET)
