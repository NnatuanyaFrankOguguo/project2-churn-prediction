# 🤖 AI Usage Documentation

This document describes how AI (GitHub Copilot / Claude) was used in developing this Customer Churn Prediction & CLV Analysis project.

## Summary

AI assistance was used as a **pair programming partner** to accelerate development while maintaining code quality and understanding. Every AI-generated component was reviewed, tested, and verified for correctness.

---

## Phase 1: Data Architecture & CLV Logic

### What AI Helped With

1. **Project Structure Design**
   - Prompt: "Create a modular project structure for a churn prediction system with separate modules for data prep, training, interpretability, and a Streamlit app"
   - AI generated the folder structure and initial file stubs

2. **TotalCharges Missing Value Strategy**
   - Prompt: "How should I handle TotalCharges missing values in the Telco dataset?"
   - AI suggested: "If tenure = 0, TotalCharges should be 0 (new customer). Otherwise, impute with MonthlyCharges."
   - **Verified:** Checked that 11 missing values all had tenure = 0

3. **Feature Engineering Ideas**
   - Prompt: "What risk flags make business sense for churn prediction?"
   - AI suggested several; I selected 3:
     - `flag_internet_no_tech_support` (unsupported customers)
     - `flag_fiber_high_charges` (premium unsatisfied)
     - `flag_short_tenure_monthly` (flight risk)

4. **CLV Formula**
   - AI helped draft the expected tenure assumptions
   - **I modified:** Changed from flat 18 months to contract-based (24m for contracts, 12m for month-to-month)

### What I Fixed/Verified

- ✅ Verified LabelEncoder alphabetical ordering matches between data_prep.py and app.py
- ✅ Confirmed stratification worked (checked churn rates in train/val/test)
- ✅ Fixed CLV_quartile being included in model features (it's only for analysis)

---

## Phase 2: Model Training

### What AI Helped With

1. **Class Imbalance Handling**
   - Prompt: "My churn rate is 26.5%. How do I handle this in XGBoost?"
   - AI suggested: `scale_pos_weight = neg_count / pos_count`
   - Also recommended `class_weight='balanced'` for sklearn models

2. **Hyperparameter Grid Selection**
   - Prompt: "What hyperparameters should I tune for Random Forest and XGBoost for churn prediction?"
   - AI suggested:
     - RF: max_depth, min_samples_leaf
     - XGB: max_depth, learning_rate
   - **I chose:** Limited grid for speed (was originally too large)

3. **High-Risk Profile Test**
   - AI helped construct the test profile dictionary matching feature encodings
   - **Bug Found:** Initial test failed because feature order didn't match - fixed by using model's feature_names

### What I Fixed/Verified

- ✅ Verified all 3 models detect high-risk profile with >60% probability
- ✅ Confirmed recall > 60% for all models (requirement)
- ✅ Checked that Contract, Tenure, MonthlyCharges appear in top feature importances

---

## Phase 3: Interpretability

### What AI Helped With

1. **SHAP Explainer Selection**
   - Prompt: "Should I use KernelExplainer for Logistic Regression?"
   - AI advised against: "Use coefficient analysis instead - faster and more interpretable for linear models"
   - Formula: `importance = |coefficient × std_dev|`

2. **SHAP Value Handling**
   - AI helped handle different SHAP output formats (list vs array for binary classification)
   - Fixed: `if isinstance(shap_values, list): shap_values = shap_values[1]`

3. **Local vs Global Explanations**
   - Prompt: "How do I create a waterfall plot for a single prediction?"
   - AI provided the bar chart implementation with color coding (red=increases churn, green=decreases)

### What I Fixed/Verified

- ✅ Fixed bug with SHAP values being numpy arrays (added `.flatten()` and float conversion)
- ✅ Verified sample size of 100-200 is sufficient for SHAP without being too slow
- ✅ Confirmed feature importance plots match expected patterns (Contract at top)

---

## Phase 4: Streamlit App

### What AI Helped With

1. **App Architecture**
   - Prompt: "Create a Streamlit app with 3 tabs for churn prediction, model performance, and business insights"
   - AI generated the tab structure and caching decorators

2. **Encoding Consistency**
   - AI helped create the ENCODING_MAPS dictionary matching Phase 1 LabelEncoder output
   - Critical for prediction accuracy!

3. **Visualization Design**
   - AI suggested color schemes (red for churn risk, green for decreases)
   - Helped with matplotlib styling for professional appearance

4. **CLV Insights Section**
   - Prompt: "What actionable recommendations should I give based on CLV × churn analysis?"
   - AI drafted recommendations; I refined for business context

### What I Fixed/Verified

- ✅ Tested prediction with high-risk profile manually
- ✅ Verified internet service dropdown correctly hides options when "No" is selected
- ✅ Fixed model key mapping (was using wrong string format for model selection)
- ✅ Tested that ensemble prediction averages all 3 models correctly

---

## Key Prompts That Mattered

### 1. Initial Project Setup
```
Role: You are a Senior ML Engineer. Build a Customer Churn & CLV Prediction System 
using the IBM Telco Dataset with detailed logging, modular code, and phase-based 
implementation.
```

### 2. Feature Engineering
```
What business-driven features should I engineer for churn prediction? 
Consider customer lifecycle, service engagement, and billing patterns.
```

### 3. Model Comparison
```
Why isn't XGBoost beating Logistic Regression? What could be wrong?
```
AI diagnosed: "With strong linear features like TotalCharges and MonthlyCharges, 
logistic regression can be surprisingly competitive. Try more feature interactions."

### 4. SHAP Performance
```
SHAP is taking 30+ seconds. How do I speed it up for Streamlit?
```
AI suggested: "Sample 100-200 rows for background data. Use @st.cache_resource for explainers."

---

## What I Learned

1. **LabelEncoder Gotcha:** Alphabetical sorting means encodings aren't intuitive (e.g., "Female"=0, "Male"=1)

2. **Class Imbalance:** Using `class_weight='balanced'` significantly improved recall from ~40% to ~80%

3. **SHAP for Linear Models:** Coefficient analysis is actually *better* than SHAP for logistic regression - faster and more interpretable

4. **CLV Business Logic:** Month-to-month customers with high monthly charges are the most at-risk - counterintuitive but makes sense (they're paying a lot without commitment)

---

## Verification Checklist

| Component | Verified? | Notes |
|-----------|-----------|-------|
| Data encoding matches app encoding | ✅ | Checked all 16 categorical mappings |
| Stratification works | ✅ | Train/val/test within 0.5% of original churn rate |
| High-risk profile detection | ✅ | All models >60% probability |
| Recall > 60% | ✅ | All models 78-80% |
| SHAP explanations work | ✅ | TreeExplainer for RF/XGB |
| App loads in < 5 seconds | ✅ | Using caching |
| Single prediction < 2 seconds | ✅ | Tested with timer |

---

## Conclusion

AI assistance accelerated development by approximately 3-4x while maintaining code quality. The key was treating AI as a **pair programmer** rather than a code generator - every suggestion was reviewed, tested, and often modified to fit the specific requirements of this project.

The most valuable AI contributions were:
1. Debugging SHAP array handling issues
2. Suggesting coefficient analysis over KernelExplainer
3. Helping construct the encoding consistency between training and prediction

Total development time: ~6 hours (estimated 20+ hours without AI)
