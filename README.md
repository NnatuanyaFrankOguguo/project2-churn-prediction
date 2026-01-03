# 🔮 Customer Churn Prediction & CLV Analysis

A production-ready machine learning system that predicts customer churn and calculates Customer Lifetime Value (CLV) to help SaaS companies prioritize retention efforts.

## 🎯 Business Problem

SaaS companies lose 5-7% of revenue annually to customer churn. This project provides:
1. **Churn Prediction** - Identify customers likely to leave before they do
2. **CLV Estimation** - Understand which customers are most valuable to retain
3. **Actionable Insights** - Know WHY customers churn and what to do about it

## 📊 Key Findings

| CLV Quartile | Churn Rate | Action |
|--------------|------------|--------|
| Low | ~18% | Monitor |
| Medium | ~35% | Engage |
| **High** | **~44%** | **🚨 Priority Retention** |
| Premium | ~10% | Nurture |

**Critical Insight:** High CLV customers have the highest churn rate! These are month-to-month customers with Fiber Optic internet but no security/support services. They're paying premium prices but feel they're not getting value.

## 🏗️ Architecture

```
project2-churn-prediction/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── AI_USAGE.md           # AI assistance documentation
├── data/
│   ├── raw/              # Original IBM Telco dataset
│   └── processed/        # Train/val/test splits + encoders
├── models/
│   ├── logistic.pkl      # Logistic Regression model
│   ├── random_forest.pkl # Random Forest model
│   ├── xgboost.pkl       # XGBoost model
│   ├── scaler.pkl        # StandardScaler for features
│   └── model_results.pkl # Evaluation metrics
└── src/
    ├── data_prep.py      # Phase 1: Data preparation & CLV
    ├── train_models.py   # Phase 2: Model training
    └── interpretability.py # Phase 3: SHAP explanations
```

## 📈 Model Performance

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.51 | **0.79** | 0.62 | **0.84** |
| Random Forest | 0.51 | 0.80 | 0.62 | 0.84 |
| XGBoost | 0.50 | 0.78 | 0.61 | 0.84 |

All models achieve:
- ✅ Recall > 60% (catching most churners)
- ✅ AUC-ROC > 0.84 (strong discrimination)
- ✅ Correctly identify high-risk profiles (>60% probability)

## 💰 CLV Calculation

**Formula:** `CLV = MonthlyCharges × ExpectedTenure`

**Assumptions:**
- **One year / Two year contracts:** Expected tenure = 24 months (committed customers)
- **Month-to-month:** Expected tenure = 12 months (higher churn risk)

**Why these assumptions?**
- Contract customers have demonstrated commitment through contractual lock-in
- Month-to-month customers can leave anytime with minimal friction
- 12-month baseline for month-to-month reflects typical churn patterns in SaaS

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/project2-churn-prediction.git
cd project2-churn-prediction
```

2. **Create virtual environment:**
```bash
# Using uv (recommended)
uv venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt
```

3. **Run data preparation (optional - data already included):**
```bash
python src/data_prep.py
```

4. **Train models (optional - models already included):**
```bash
python src/train_models.py
```

5. **Launch the app:**
```bash
streamlit run app.py
```

### Deployed App

🔗 **Live Demo:** [Your Streamlit Cloud URL]

## 🎥 Video Demo

📹 **Watch the demo:** [YouTube/Loom Link]

**Demo covers:**
- 0:00-0:30 - Business problem & value proposition
- 0:30-1:00 - Live prediction with a high-risk customer
- 1:00-2:00 - SHAP explanations & model comparison
- 2:00-2:30 - CLV insights & recommendations

## 🔬 Technical Details

### Feature Engineering

| Feature | Description | Business Logic |
|---------|-------------|----------------|
| `tenure_bucket` | 0-6m, 6-12m, 12-24m, 24m+ | Customer lifecycle stage |
| `services_count` | Total subscribed services | Service engagement level |
| `monthly_to_total_ratio` | Spending pattern indicator | Value vs. expectations |
| `flag_internet_no_tech_support` | High-risk combination | Unsupported premium users |
| `flag_fiber_high_charges` | Fiber + high monthly cost | Premium but potentially unsatisfied |
| `flag_short_tenure_monthly` | New + no commitment | Highest flight risk |

### Interpretability

- **XGBoost & Random Forest:** SHAP TreeExplainer for both global and local explanations
- **Logistic Regression:** Standardized coefficients (|coef × std_dev|) - faster and more interpretable for linear models

### Class Imbalance Handling

- Churn rate: ~26.5% (imbalanced)
- **Logistic/RF:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight=2.77` (ratio of majority to minority)

## 📋 Data Source

[IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)

- 7,043 customers
- 21 original features (demographics, services, billing)
- Binary target: Churn (Yes/No)

## 🤖 AI Assistance

See [AI_USAGE.md](AI_USAGE.md) for detailed documentation on how AI was used in this project.

## 📝 License

MIT License - feel free to use for educational purposes.

---

*Built as part of a Data Science portfolio project demonstrating end-to-end ML system development.*
