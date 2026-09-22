# Employee Attrition Prediction

> **Predict whether an employee will leave (Attrition = Yes/No) using Machine Learning — built with Python, Streamlit, and Scikit-learn.**

---

## Project Objective

Organizations face significant operational and financial costs when employees resign unexpectedly. This project analyses IBM HR Analytics data and builds three Machine Learning models to predict employee attrition based on factors such as **demographics, job role, salary, overtime, job satisfaction, work-life balance, and tenure**.

---

## Dataset Information

| Property        | Value |
|----------------|-------|
| **Source**     | IBM HR Analytics Employee Attrition & Performance |
| **File**       | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| **Rows**       | 1,470 employees |
| **Columns**    | 35 features |
| **Target**     | `Attrition` (Yes = 1, No = 0) |
| **Class split**| ~83.9% No (Stay) · ~16.1% Yes (Leave) |

### Key Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Employee age |
| MonthlyIncome | Numeric | Monthly salary in USD |
| OverTime | Categorical | Whether employee works overtime (Yes/No) |
| JobLevel | Numeric | Seniority level (1–5) |
| TotalWorkingYears | Numeric | Total years of work experience |
| YearsAtCompany | Numeric | Tenure at this company |
| JobSatisfaction | Numeric | Satisfaction score (1–4) |
| EnvironmentSatisfaction | Numeric | Workplace satisfaction (1–4) |
| WorkLifeBalance | Numeric | Work-life balance score (1–4) |
| BusinessTravel | Categorical | Travel frequency |
| MaritalStatus | Categorical | Single / Married / Divorced |
| Department | Categorical | Sales / R&D / HR |
| DistanceFromHome | Numeric | Commute distance in km |

> **Removed columns** (constant/ID — zero predictive value): `EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core language |
| **Streamlit 1.64** | Web frontend / dashboard |
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib / Seaborn** | Data visualisation |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **imbalanced-learn** | SMOTE for class imbalance |

---

## Project Structure

```
ibm_project/
│
├── app.py                              ← Main Streamlit application (all logic)
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← Dataset (must be in same folder)
├── requirements.txt                    ← Python dependencies
├── README.md                           ← This file
└── PROJECT_REPORT.md                   ← Full professional report
```

---

## Installation

### Prerequisites
- Python **3.9 or higher** installed
- `pip` available in terminal

### Step 1 — Clone or download the project

```bash
# If using git:
git clone <repo-url>
cd ibm_project

# Or simply download and unzip the project folder
```

### Step 2 — (Optional but recommended) Create a virtual environment

```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — macOS / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** in your default browser.

> **Note:** Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the **same folder** as `app.py`.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Overview** | KPI cards, attrition distribution chart, dataset summary, ML pipeline steps |
| **🔍 EDA** | Numeric histograms, categorical attrition rates, deep-dive selectors, correlation chart |
| **📈 Model Performance** | Metrics table, bar comparison, confusion matrices, ROC curves, feature importances, classification reports |
| **🔮 Predict Attrition** | Employee details form → Yes/No prediction + probability gauge + all-model vote + top influencing factors |
| **💡 Insights & Recommendations** | Key findings, HR action recommendations, limitations, conclusion |

---

## ML Models Trained

| Model | Notes |
|-------|-------|
| **Logistic Regression** | Baseline linear model; trained on SMOTE + StandardScaler data |
| **Decision Tree** | Non-linear tree (max_depth=6); trained on SMOTE data |
| **Random Forest** | Ensemble of 300 trees (max_depth=10); best ROC-AUC |

### Results (held-out test set — 20%, stratified)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | ~0.799 | ~0.449 | ~0.511 | ~0.449 | ~0.718 |
| Decision Tree       | ~0.745 | ~0.348 | ~0.426 | ~0.348 | ~0.645 |
| Random Forest       | ~0.799 | ~0.379 | ~0.383 | ~0.379 | ~0.727 |

> Results are generated **live from the actual dataset** — never hardcoded.

---

## Key Findings

- **OverTime** is the strongest predictor of attrition (~3× higher risk)
- **Low Monthly Income** (below median) significantly raises attrition risk
- **Single employees** leave ~2× more than married employees
- **Sales Representatives** and **Laboratory Technicians** are highest-risk roles
- **Distance from home > 20 km** correlates with elevated attrition
- Low **Job Satisfaction** and **Environment Satisfaction** scores (1–2) are strong warning signals

---

## How Prediction Works

1. User fills in the **Employee Prediction Form** (30 fields)
2. Categorical inputs are encoded using the **same LabelEncoders** fitted during training
3. The **Random Forest** model (best ROC-AUC) outputs a **probability score**
4. Result displayed: **Attrition = Yes / No**, probability gauge, all-model votes, and top 8 influencing features

---

## License

This project uses the publicly available IBM HR Analytics dataset for educational purposes.
