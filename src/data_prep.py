"""
Data Preparation Module for Customer Churn Prediction System
=============================================================
This module handles:
1. Loading and cleaning the IBM Telco Customer Churn dataset
2. Feature engineering (tenure buckets, service counts, risk flags)
3. CLV (Customer Lifetime Value) calculation
4. Data splitting with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from typing import Tuple, Dict

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


class ChurnDataPreparation:
    """
    Handles all data preparation steps for the Churn Prediction system.
    
    This class follows a phase-based approach with detailed logging at each step.
    """
    
    def __init__(self, raw_data_path: str, output_dir: str):
        """
        Initialize the data preparation pipeline.
        
        Args:
            raw_data_path: Path to the raw CSV file
            output_dir: Directory to save processed data
        """
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.df = None
        self.label_encoders = {}
        self.scaler = None
        
        log_info(f"ChurnDataPreparation initialized")
        log_info(f"  └─ Raw data path: {raw_data_path}")
        log_info(f"  └─ Output directory: {output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        STEP 1.1: Load the IBM Telco Customer Churn dataset
        
        Returns:
            Loaded DataFrame
        """
        log_info("=" * 70)
        log_info("STEP 1.1: Loading IBM Telco Customer Churn Dataset")
        log_info("=" * 70)
        
        try:
            self.df = pd.read_csv(self.raw_data_path)
            log_success(f"Dataset loaded successfully")
            log_info(f"  └─ Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            log_info(f"  └─ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Log column types
            log_info(f"  └─ Column types breakdown:")
            log_info(f"      • Numeric: {len(self.df.select_dtypes(include=[np.number]).columns)}")
            log_info(f"      • Object: {len(self.df.select_dtypes(include=['object']).columns)}")
            
            return self.df
            
        except FileNotFoundError as e:
            log_error(
                "Failed to load dataset",
                root_cause=f"File not found: {self.raw_data_path}",
                location="ChurnDataPreparation.load_data()"
            )
            raise
        except Exception as e:
            log_error(
                "Failed to load dataset",
                root_cause=str(e),
                location="ChurnDataPreparation.load_data()"
            )
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        STEP 1.2: Clean and preprocess the dataset
        
        Key operations:
        - Handle TotalCharges missing/invalid values
        - Convert data types appropriately
        - Remove customerID (not a feature)
        
        Returns:
            Cleaned DataFrame
        """
        log_info("=" * 70)
        log_info("STEP 1.2: Data Cleaning & Type Conversion")
        log_info("=" * 70)
        
        # Check for missing values
        log_info("Checking for missing values...")
        missing_counts = self.df.isnull().sum()
        if missing_counts.sum() > 0:
            log_warning(f"Found {missing_counts.sum()} missing values across {(missing_counts > 0).sum()} columns")
            for col in missing_counts[missing_counts > 0].index:
                log_warning(f"  └─ {col}: {missing_counts[col]} missing ({missing_counts[col]/len(self.df)*100:.2f}%)")
        else:
            log_success("No explicit missing values found")
        
        # Handle TotalCharges (known issue: sometimes stored as string with spaces)
        log_info("Processing TotalCharges column...")
        original_dtype = self.df['TotalCharges'].dtype
        log_info(f"  └─ Original dtype: {original_dtype}")
        
        if self.df['TotalCharges'].dtype == 'object':
            log_warning("TotalCharges is stored as object type, converting to numeric")
            
            # Convert to numeric, coercing errors to NaN
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            
            # Count how many were converted to NaN
            nan_count = self.df['TotalCharges'].isnull().sum()
            if nan_count > 0:
                log_warning(f"Found {nan_count} non-numeric TotalCharges values converted to NaN")
                
                # Strategy: Fill with 0 for new customers (tenure = 0) or MonthlyCharges for others
                log_info("Imputation strategy:")
                log_info("  └─ If tenure = 0: TotalCharges = 0 (new customer)")
                log_info("  └─ Otherwise: TotalCharges = MonthlyCharges (first month)")
                
                mask_missing = self.df['TotalCharges'].isnull()
                mask_new_customer = (self.df['tenure'] == 0)
                
                # New customers get 0
                self.df.loc[mask_missing & mask_new_customer, 'TotalCharges'] = 0
                
                # Others get their monthly charge (approximation for first month)
                self.df.loc[mask_missing & ~mask_new_customer, 'TotalCharges'] = \
                    self.df.loc[mask_missing & ~mask_new_customer, 'MonthlyCharges']
                
                log_success(f"Imputed {nan_count} TotalCharges values")
        
        log_success(f"TotalCharges now has dtype: {self.df['TotalCharges'].dtype}")
        
        # Convert Churn to binary
        log_info("Converting Churn column to binary (0/1)...")
        if self.df['Churn'].dtype == 'object':
            self.df['Churn'] = (self.df['Churn'] == 'Yes').astype(int)
            log_success(f"Churn converted: Yes→1, No→0")
            churn_rate = self.df['Churn'].mean()
            log_info(f"  └─ Churn rate: {churn_rate*100:.2f}% ({self.df['Churn'].sum()} out of {len(self.df)})")
            
            if churn_rate < 0.15 or churn_rate > 0.40:
                log_warning(f"Churn rate is {churn_rate*100:.2f}%, indicating class imbalance")
        
        # Remove customerID (not a feature)
        if 'customerID' in self.df.columns:
            log_info("Removing customerID column (identifier, not a feature)")
            self.df = self.df.drop('customerID', axis=1)
            log_success("customerID removed")
        
        log_success(f"Data cleaning complete. Final shape: {self.df.shape}")
        
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        STEP 1.3: Feature Engineering
        
        Creates business-driven features:
        1. tenure_bucket: Categorical tenure grouping
        2. services_count: Total number of services subscribed
        3. monthly_to_total_ratio: Spending pattern indicator
        4. Risk flags: High-risk customer combinations
        
        Returns:
            DataFrame with engineered features
        """
        log_info("=" * 70)
        log_info("STEP 1.3: Feature Engineering")
        log_info("=" * 70)
        
        # 1. Tenure Buckets
        log_info("Creating tenure_bucket feature...")
        log_info("  └─ Buckets: 0-6m, 6-12m, 12-24m, 24m+")
        
        def categorize_tenure(tenure):
            """Categorize tenure into business-meaningful buckets"""
            if tenure <= 6:
                return '0-6m'
            elif tenure <= 12:
                return '6-12m'
            elif tenure <= 24:
                return '12-24m'
            else:
                return '24m+'
        
        self.df['tenure_bucket'] = self.df['tenure'].apply(categorize_tenure)
        log_success("tenure_bucket created")
        log_info(f"  └─ Distribution:\n{self.df['tenure_bucket'].value_counts().to_string()}")
        
        # 2. Services Count
        log_info("Creating services_count feature...")
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count services (excluding "No internet service" and "No phone service" as these are N/A)
        services_count = 0
        for col in service_columns:
            if col in self.df.columns:
                # Count as a service if it's "Yes" or the service type itself (for InternetService)
                if col == 'InternetService':
                    services_count += (self.df[col].isin(['DSL', 'Fiber optic'])).astype(int)
                elif col == 'PhoneService':
                    services_count += (self.df[col] == 'Yes').astype(int)
                else:
                    services_count += (self.df[col] == 'Yes').astype(int)
        
        self.df['services_count'] = services_count
        log_success(f"services_count created")
        log_info(f"  └─ Range: {self.df['services_count'].min()} to {self.df['services_count'].max()}")
        log_info(f"  └─ Mean: {self.df['services_count'].mean():.2f}")
        
        # 3. Monthly to Total Ratio
        log_info("Creating monthly_to_total_ratio feature...")
        log_info("  └─ Formula: TotalCharges / max(1, tenure × MonthlyCharges)")
        
        expected_total = self.df['tenure'] * self.df['MonthlyCharges']
        expected_total = expected_total.replace(0, 1)  # Avoid division by zero
        self.df['monthly_to_total_ratio'] = self.df['TotalCharges'] / expected_total
        
        log_success("monthly_to_total_ratio created")
        log_info(f"  └─ Range: {self.df['monthly_to_total_ratio'].min():.3f} to {self.df['monthly_to_total_ratio'].max():.3f}")
        
        # Check for anomalies
        anomaly_mask = (self.df['monthly_to_total_ratio'] > 1.5) | (self.df['monthly_to_total_ratio'] < 0.5)
        anomaly_count = anomaly_mask.sum()
        if anomaly_count > 0:
            log_warning(f"Found {anomaly_count} customers with unusual spending patterns (ratio < 0.5 or > 1.5)")
        
        # 4. Risk Flags
        log_info("Creating risk flag features...")
        
        # Flag: Internet but no tech support (high churn risk)
        self.df['flag_internet_no_tech_support'] = (
            (self.df['InternetService'].isin(['DSL', 'Fiber optic'])) &
            (self.df['TechSupport'] == 'No')
        ).astype(int)
        
        # Flag: Fiber optic with high charges (premium unsatisfied customer risk)
        self.df['flag_fiber_high_charges'] = (
            (self.df['InternetService'] == 'Fiber optic') &
            (self.df['MonthlyCharges'] > self.df['MonthlyCharges'].quantile(0.75))
        ).astype(int)
        
        # Flag: Short tenure with month-to-month contract (highest churn risk)
        self.df['flag_short_tenure_monthly'] = (
            (self.df['tenure'] <= 12) &
            (self.df['Contract'] == 'Month-to-month')
        ).astype(int)
        
        log_success("Created 3 risk flag features:")
        log_info(f"  └─ flag_internet_no_tech_support: {self.df['flag_internet_no_tech_support'].sum()} customers")
        log_info(f"  └─ flag_fiber_high_charges: {self.df['flag_fiber_high_charges'].sum()} customers")
        log_info(f"  └─ flag_short_tenure_monthly: {self.df['flag_short_tenure_monthly'].sum()} customers")
        
        log_success(f"Feature engineering complete. New shape: {self.df.shape}")
        
        return self.df
    
    def calculate_clv(self) -> pd.DataFrame:
        """
        Calculate Customer Lifetime Value (CLV)
        
        Formula: CLV = MonthlyCharges × ExpectedTenure
        
        Expected Tenure Assumptions:
        - One year / Two year contracts: 24 months (stable, committed customers)
        - Month-to-month: 12 months (higher churn risk)
        
        Returns:
            DataFrame with CLV and CLV quartile segments
        """
        log_info("=" * 70)
        log_info("STEP 1.4: Customer Lifetime Value (CLV) Calculation")
        log_info("=" * 70)
        
        log_info("Defining Expected Tenure based on Contract type:")
        log_info("  └─ One year / Two year contracts: 24 months (committed)")
        log_info("  └─ Month-to-month: 12 months (higher churn)")
        
        # Define expected tenure based on contract type
        def get_expected_tenure(contract_type):
            """Determine expected tenure based on contract type"""
            if contract_type in ['One year', 'Two year']:
                return 24  # Stable customers
            else:  # Month-to-month
                return 12  # Higher risk
        
        self.df['expected_tenure'] = self.df['Contract'].apply(get_expected_tenure)
        
        # Calculate CLV
        log_info("Calculating CLV = MonthlyCharges × ExpectedTenure...")
        self.df['CLV'] = self.df['MonthlyCharges'] * self.df['expected_tenure']
        
        log_success("CLV calculated for all customers")
        log_info(f"  └─ CLV Range: ${self.df['CLV'].min():.2f} to ${self.df['CLV'].max():.2f}")
        log_info(f"  └─ CLV Mean: ${self.df['CLV'].mean():.2f}")
        log_info(f"  └─ CLV Median: ${self.df['CLV'].median():.2f}")
        log_info(f"  └─ Total CLV: ${self.df['CLV'].sum():,.2f}")
        
        # Create CLV Quartiles
        log_info("Segmenting customers into CLV Quartiles...")
        self.df['CLV_quartile'] = pd.qcut(
            self.df['CLV'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        log_success("CLV Quartiles created")
        log_info("  └─ Distribution:")
        for quartile in ['Low', 'Medium', 'High', 'Premium']:
            count = (self.df['CLV_quartile'] == quartile).sum()
            mean_clv = self.df[self.df['CLV_quartile'] == quartile]['CLV'].mean()
            log_info(f"      • {quartile}: {count} customers (avg CLV: ${mean_clv:.2f})")
        
        # Analyze churn rate by CLV quartile
        log_info("Analyzing Churn Rate by CLV Quartile:")
        for quartile in ['Low', 'Medium', 'High', 'Premium']:
            mask = self.df['CLV_quartile'] == quartile
            churn_rate = self.df[mask]['Churn'].mean()
            count = mask.sum()
            log_info(f"  └─ {quartile}: {churn_rate*100:.2f}% churn ({count} customers)")
        
        log_success("CLV analysis complete")
        
        return self.df
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder
        
        IMPORTANT: LabelEncoder sorts alphabetically!
        - Gender: Female=0, Male=1
        - MultipleLines: No=0, No Phone Service=1, Yes=2
        
        Returns:
            DataFrame with encoded categorical variables
        """
        log_info("=" * 70)
        log_info("STEP 1.5: Encoding Categorical Features")
        log_info("=" * 70)
        
        log_warning("LabelEncoder sorts alphabetically - encoding must match in app!")
        
        # Identify categorical columns (exclude target and already created features)
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove CLV_quartile from encoding (we'll keep it for analysis)
        if 'CLV_quartile' in categorical_cols:
            categorical_cols.remove('CLV_quartile')
        
        log_info(f"Found {len(categorical_cols)} categorical columns to encode")
        
        # Encode each categorical column
        for col in categorical_cols:
            log_info(f"Encoding {col}...")
            
            # Show unique values before encoding
            unique_vals = sorted(self.df[col].unique())
            log_info(f"  └─ Unique values: {unique_vals}")
            
            # Create and fit label encoder
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            
            # Store encoder for later use
            self.label_encoders[col] = le
            
            # Show encoding mapping
            log_info(f"  └─ Encoding map:")
            for idx, val in enumerate(le.classes_):
                log_info(f"      • {val} → {idx}")
            
            log_success(f"{col} encoded")
        
        log_success(f"All categorical features encoded. Total encoders saved: {len(self.label_encoders)}")
        
        return self.df
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Split data into train/validation/test sets with stratification
        
        Split ratio: 60% train, 20% validation, 20% test
        Stratification on 'Churn' to maintain class balance
        
        Args:
            test_size: Proportion for test set (default: 0.2)
            val_size: Proportion for validation set from remaining data (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        log_info("=" * 70)
        log_info("STEP 1.6: Data Splitting with Stratification")
        log_info("=" * 70)
        
        log_info(f"Target split ratio: 60% train, 20% validation, 20% test")
        log_info(f"Stratification on 'Churn' column to maintain class balance")
        log_info(f"Random state: {random_state}")
        
        # Separate features and target
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        log_info(f"Dataset size: {len(self.df)} samples")
        log_info(f"  └─ Positive class (Churn=1): {y.sum()} ({y.mean()*100:.2f}%)")
        log_info(f"  └─ Negative class (Churn=0): {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
        
        # First split: separate test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        log_success(f"Test set created: {len(X_test)} samples ({len(X_test)/len(self.df)*100:.1f}%)")
        
        # Second split: separate validation set from remaining (20% of total = 25% of temp)
        val_size_adjusted = val_size / (1 - test_size)  # Adjust to get 20% of original
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_state
        )
        
        log_success(f"Validation set created: {len(X_val)} samples ({len(X_val)/len(self.df)*100:.1f}%)")
        log_success(f"Training set created: {len(X_train)} samples ({len(X_train)/len(self.df)*100:.1f}%)")
        
        # Verify stratification
        log_info("Verifying stratification (Churn rate in each set):")
        train_churn = y_train.mean()
        val_churn = y_val.mean()
        test_churn = y_test.mean()
        original_churn = y.mean()
        
        log_info(f"  └─ Original: {original_churn*100:.2f}%")
        log_info(f"  └─ Train:    {train_churn*100:.2f}% (diff: {abs(train_churn-original_churn)*100:.2f}%)")
        log_info(f"  └─ Val:      {val_churn*100:.2f}% (diff: {abs(val_churn-original_churn)*100:.2f}%)")
        log_info(f"  └─ Test:     {test_churn*100:.2f}% (diff: {abs(test_churn-original_churn)*100:.2f}%)")
        
        # Check if stratification is good (within 2% difference)
        max_diff = max(
            abs(train_churn - original_churn),
            abs(val_churn - original_churn),
            abs(test_churn - original_churn)
        ) * 100
        
        if max_diff < 2.0:
            log_success("Stratification verified: All splits within 2% of original distribution")
        else:
            log_warning(f"Stratification deviation: {max_diff:.2f}% (acceptable if < 5%)")
        
        # Combine back into full DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        log_success("Data splitting complete")
        
        return splits
    
    def save_processed_data(self, splits: Dict):
        """
        Save processed data splits and preprocessing artifacts
        
        Saves:
        - train.csv, val.csv, test.csv (processed data)
        - label_encoders.pkl (for app to use)
        - feature_names.pkl (for reference)
        
        Args:
            splits: Dictionary containing train/val/test DataFrames
        """
        log_info("=" * 70)
        log_info("STEP 1.7: Saving Processed Data")
        log_info("=" * 70)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save each split
        for split_name, df in splits.items():
            output_path = os.path.join(self.output_dir, f"{split_name}.csv")
            df.to_csv(output_path, index=False)
            log_success(f"Saved {split_name}.csv ({len(df)} rows, {df.shape[1]} columns)")
            log_info(f"  └─ Path: {output_path}")
        
        # Save label encoders
        encoders_path = os.path.join(self.output_dir, "label_encoders.pkl")
        joblib.dump(self.label_encoders, encoders_path)
        log_success(f"Saved label_encoders.pkl ({len(self.label_encoders)} encoders)")
        
        # Save feature names for reference
        feature_names = [col for col in splits['train'].columns if col != 'Churn']
        features_path = os.path.join(self.output_dir, "feature_names.pkl")
        joblib.dump(feature_names, features_path)
        log_success(f"Saved feature_names.pkl ({len(feature_names)} features)")
        
        log_success("All processed data and artifacts saved successfully")
        log_info(f"  └─ Output directory: {self.output_dir}")
        
    def run_phase1(self):
        """
        Execute the complete Phase 1 pipeline
        
        This is the main entry point that runs all Phase 1 steps in sequence:
        1. Load data
        2. Clean data
        3. Engineer features
        4. Calculate CLV
        5. Encode categorical features
        6. Split data
        7. Save processed data
        """
        log_info(f"\n{LogColors.BOLD}{'='*70}{LogColors.RESET}")
        log_info(f"{LogColors.BOLD}PHASE 1: DATA ARCHITECTURE & CLV LOGIC{LogColors.RESET}")
        log_info(f"{LogColors.BOLD}{'='*70}{LogColors.RESET}\n")
        
        try:
            # Execute all steps in sequence
            self.load_data()
            self.clean_data()
            self.engineer_features()
            self.calculate_clv()
            self.encode_categorical_features()
            splits = self.split_data()
            self.save_processed_data(splits)
            
            # Final summary
            log_info("\n" + "=" * 70)
            log_success("PHASE 1 COMPLETE: Data Architecture & CLV Logic")
            log_info("=" * 70)
            log_info("Summary:")
            log_info(f"  └─ Total samples: {len(self.df)}")
            log_info(f"  └─ Total features: {len([col for col in self.df.columns if col != 'Churn'])}")
            log_info(f"  └─ Train samples: {len(splits['train'])}")
            log_info(f"  └─ Val samples: {len(splits['val'])}")
            log_info(f"  └─ Test samples: {len(splits['test'])}")
            log_info(f"  └─ Churn rate: {self.df['Churn'].mean()*100:.2f}%")
            log_info(f"  └─ Average CLV: ${self.df['CLV'].mean():.2f}")
            log_success(f"All data saved to: {self.output_dir}")
            log_info("=" * 70 + "\n")
            
            return splits
            
        except Exception as e:
            log_error(
                "PHASE 1 FAILED",
                root_cause=str(e),
                location="ChurnDataPreparation.run_phase1()"
            )
            raise


if __name__ == "__main__":
    """
    Main execution block for Phase 1
    """
    # Define paths
    RAW_DATA_PATH = r"c:\project2-churn-prediction\data\raw\Telco-Customer-Churn.csv"
    OUTPUT_DIR = r"c:\project2-churn-prediction\data\processed"
    
    # Initialize and run Phase 1
    prep = ChurnDataPreparation(RAW_DATA_PATH, OUTPUT_DIR)
    splits = prep.run_phase1()
    
    print("\n" + LogColors.GREEN + LogColors.BOLD + "✓ Phase 1 execution completed successfully!" + LogColors.RESET)
    print(LogColors.BLUE + "→ Ready to proceed to Phase 2: Model Training" + LogColors.RESET)
