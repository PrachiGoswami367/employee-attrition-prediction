# PROJECT REPORT
## Employee Attrition Prediction Using Machine Learning

---

| Field | Detail |
|-------|--------|
| **Project Title** | Employee Attrition Prediction Using Machine Learning |
| **Dataset** | IBM HR Analytics Employee Attrition & Performance |
| **File** | `WA_Fn-UseC_-HR-Employee-Attrition.csv` |
| **Tools** | Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · imbalanced-learn |
| **Models** | Logistic Regression · Decision Tree · Random Forest |
| **Target Variable** | `Attrition` (Yes = 1 / No = 0) |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Dataset Description](#4-dataset-description)
5. [Data Cleaning](#5-data-cleaning)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Feature Engineering & Preprocessing](#7-feature-engineering--preprocessing)
8. [ML Pipeline](#8-ml-pipeline)
9. [Model Evaluation & Results](#9-model-evaluation--results)
10. [Feature Importance](#10-feature-importance)
11. [UI Screenshot — Overview Page](#11-ui-screenshot--overview-page)
12. [UI Screenshot — EDA Page](#12-ui-screenshot--eda-page)
13. [UI Screenshot — Model Performance Page](#13-ui-screenshot--model-performance-page)
14. [UI Screenshot — Predict Attrition Page](#14-ui-screenshot--predict-attrition-page)
15. [UI Screenshot — Insights Page](#15-ui-screenshot--insights-page)
16. [Prediction Results](#16-prediction-results)
17. [Key Findings](#17-key-findings)
18. [Business Recommendations](#18-business-recommendations)
19. [Limitations](#19-limitations)
20. [Conclusion](#20-conclusion)
21. [How to Run](#21-how-to-run)

---

## 1. Introduction

Employee attrition — the voluntary or involuntary departure of staff — is one of the most costly challenges in Human Resource Management. Studies estimate the cost of replacing a single employee at **50–200% of their annual salary**, when recruitment, onboarding, lost productivity, and knowledge transfer are accounted for.

This project delivers an **end-to-end Machine Learning solution** built entirely in Python using Streamlit for the frontend. It analyses the IBM HR Analytics dataset, identifies key attrition drivers, trains and compares three ML classifiers, and provides a live interactive dashboard for HR teams.

---

## 2. Problem Statement

> *Organizations face significant challenges when employees leave unexpectedly. This project analyses employee data and builds a Machine Learning model to predict whether an employee is likely to leave (**Attrition = Yes/No**) based on factors such as demographics, job role, salary, overtime, job satisfaction, work-life balance, and tenure.*

---

## 3. Objectives

1. Perform **data cleaning** to remove irrelevant and constant-value columns
2. Conduct **Exploratory Data Analysis (EDA)** to understand feature distributions and attrition patterns
3. Handle **class imbalance** (84% No / 16% Yes) using SMOTE on the training set only
4. **Label-encode** categorical features and **scale** numeric features appropriately
5. Train and compare **Logistic Regression, Decision Tree, and Random Forest**
6. Evaluate using **Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix**
7. Identify the **most important features** driving attrition predictions
8. Deploy an interactive **Streamlit dashboard** with a live employee prediction form

---

## 4. Dataset Description

| Property | Value |
|----------|-------|
| Source | IBM HR Analytics Employee Attrition & Performance |
| Total Rows | 1,470 employees |
| Total Columns | 35 features |
| Target Column | `Attrition` (Yes / No) |
| Missing Values | **None** |
| Duplicate Rows | **None** |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| No (Stay) | 1,233 | 83.9% |
| Yes (Leave) | 237 | 16.1% |

> ⚠️ The dataset is **significantly imbalanced**. SMOTE is applied to the training set to address this.

### Feature Categories

| Category | Features |
|----------|---------|
| Demographic | Age, Gender, MaritalStatus, Education, EducationField |
| Job | Department, JobRole, JobLevel, BusinessTravel, OverTime, JobInvolvement |
| Compensation | MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel |
| Experience | TotalWorkingYears, NumCompaniesWorked, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, TrainingTimesLastYear |
| Satisfaction | JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, PerformanceRating |
| **Target** | **Attrition (Yes / No)** |

---

## 5. Data Cleaning

| Step | Action | Column(s) | Reason |
|------|--------|-----------|--------|
| 1 | Remove constant columns | `EmployeeCount`, `Over18`, `StandardHours` | Single unique value — zero predictive information |
| 2 | Remove ID column | `EmployeeNumber` | Unique per row — causes data leakage |
| 3 | Check missing values | All columns | None found — dataset is complete |
| 4 | Check duplicates | All rows | None found |

**Result after cleaning:** 1,470 rows × 31 columns (30 features + 1 target). No imputation required.

---

## 6. Exploratory Data Analysis

### Numeric Feature Means by Attrition (actual dataset values)

| Feature | No (Stay) Mean | Yes (Leave) Mean | Insight |
|---------|---------------|------------------|---------|
| Age | 37.6 | 33.6 | Younger employees leave more |
| MonthlyIncome | $6,833 | $4,787 | Lower pay raises attrition risk |
| TotalWorkingYears | 11.9 | 8.2 | Less experience = higher risk |
| YearsAtCompany | 7.4 | 5.1 | Shorter tenure = higher risk |
| DistanceFromHome | 8.9 km | 10.6 km | Longer commute = higher risk |
| JobLevel | 2.14 | 1.64 | Junior roles leave more |

### Categorical Attrition Rates (actual dataset values)

| Feature | High-Risk Category | Attrition Rate | Low-Risk |
|---------|-------------------|----------------|---------|
| OverTime | Yes | ~30.5% | No: ~10.4% |
| MaritalStatus | Single | ~25.5% | Married: ~12.5% |
| BusinessTravel | Travel_Frequently | ~24.9% | Non-Travel: ~8.0% |
| JobRole | Sales Representative | ~39.8% | Manager: ~5.1% |
| Department | Sales | ~20.6% | R&D: ~13.8% |

### Top Correlations with Attrition (Pearson r — actual values)

| Feature | r Value | Direction |
|---------|---------|-----------|
| OverTime | +0.2461 | Raises risk |
| MaritalStatus | +0.1621 | Raises risk |
| TotalWorkingYears | −0.1711 | Lowers risk |
| JobLevel | −0.1691 | Lowers risk |
| MonthlyIncome | −0.1598 | Lowers risk |
| Age | −0.1592 | Lowers risk |
| StockOptionLevel | −0.1371 | Lowers risk |

---

## 7. Feature Engineering & Preprocessing

| Step | Method | Applied To | Purpose |
|------|--------|-----------|---------|
| Label Encoding | `sklearn.LabelEncoder` | 7 categorical columns | Convert text to numeric; same encoder reused at prediction |
| Stratified Split | 80/20 train/test | Full dataset | Preserve 16.1% attrition ratio in both sets |
| SMOTE | `imblearn.SMOTE` | Training set **only** | Balance 84/16 to 50/50 — prevents model bias |
| StandardScaler | `sklearn.StandardScaler` | LR train+test only | Normalise feature scale for Logistic Regression |

**SMOTE result (actual numbers):**

| Set | No | Yes | Total |
|-----|-----|-----|-------|
| Before SMOTE (train) | 986 | 190 | 1,176 |
| After SMOTE (train) | 986 | 986 | 1,972 |
| Test set (unchanged) | 247 | 47 | 294 |

---

## 8. ML Pipeline

```
1. Load CSV (1,470 rows × 35 columns)
2. Remove 4 constant/ID columns
3. Encode target: Yes → 1, No → 0
4. Label-encode 7 categorical columns
5. Stratified 80/20 train/test split
6. Apply SMOTE to training set only → 1,972 balanced samples
7. StandardScaler → fit on SMOTE data, transform for Logistic Regression
8. Train: Logistic Regression | Decision Tree (depth=6) | Random Forest (300 trees)
9. Evaluate on held-out test set (294 samples)
```

---

## 9. Model Evaluation & Results

> All metrics computed on the **held-out test set (294 samples, 20% stratified)**. No values are hardcoded.

### Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7993 | 0.4000 | **0.5106** ★ | 0.4486 | 0.7182 |
| Decision Tree | 0.7585 | 0.3235 | 0.4681 | 0.3826 | 0.6759 |
| **Random Forest** ✓ | 0.7891 | 0.3585 | 0.4043 | 0.3800 | **0.7348** ★ |

- ✅ **Random Forest** → Best **ROC-AUC (0.7348)** — used as primary prediction model
- ✅ **Logistic Regression** → Best **Recall (0.5106)** — catches the most true leavers

### Confusion Matrices (actual test set values)

| | Logistic Regression | Decision Tree | Random Forest |
|-|---------------------|---------------|---------------|
| TN (correct stays) | 211 | 201 | 213 |
| FP (false alarms) | 36 | 46 | 34 |
| FN (missed leavers) | 23 | 25 | 28 |
| TP (caught leavers) | 24 | 22 | 19 |

> ⚠️ **False Negatives (FN)** are the most costly — employees who left but were predicted to stay.

---

## 10. Feature Importance

Top 15 features by Random Forest Gini importance (actual values from trained model):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | StockOptionLevel | 0.1241 |
| 2 | JobSatisfaction | 0.0757 |
| 3 | EnvironmentSatisfaction | 0.0578 |
| 4 | MonthlyIncome | 0.0549 |
| 5 | JobInvolvement | 0.0514 |
| 6 | YearsWithCurrManager | 0.0507 |
| 7 | Age | 0.0428 |
| 8 | JobLevel | 0.0425 |
| 9 | TotalWorkingYears | 0.0395 |
| 10 | YearsInCurrentRole | 0.0393 |
| 11 | WorkLifeBalance | 0.0356 |
| 12 | YearsAtCompany | 0.0350 |
| 13 | MonthlyRate | 0.0339 |
| 14 | RelationshipSatisfaction | 0.0307 |
| 15 | DailyRate | 0.0305 |

> **StockOptionLevel** is the #1 predictor (12.41%) — employees with no stock options leave far more often.

---

## 11. Prediction Results

Two real predictions from the trained Random Forest model:

### High-Risk Employee

| Field | Value |
|-------|-------|
| Age | 26 |
| Department | Sales |
| JobRole | Sales Representative |
| JobLevel | 1 (Entry) |
| OverTime | **Yes** |
| MonthlyIncome | **$2,200** |
| StockOptionLevel | **0** |
| JobSatisfaction | **1/4** |
| WorkLifeBalance | **1/4** |
| MaritalStatus | Single |
| BusinessTravel | Travel_Frequently |

**Result → ATTRITION: YES**

| Model | Probability | Prediction |
|-------|-------------|-----------|
| Random Forest | **95.54%** | YES |
| Logistic Regression | **99.98%** | YES |
| Decision Tree | **98.16%** | YES |

---

### Low-Risk Employee

| Field | Value |
|-------|-------|
| Age | 45 |
| Department | Research & Development |
| JobRole | Manager |
| JobLevel | 4 (Senior) |
| OverTime | **No** |
| MonthlyIncome | **$14,000** |
| StockOptionLevel | **2** |
| JobSatisfaction | **4/4** |
| WorkLifeBalance | **4/4** |
| MaritalStatus | Married |
| BusinessTravel | Non-Travel |

**Result → ATTRITION: NO**

| Model | Probability | Prediction |
|-------|-------------|-----------|
| Random Forest | **3.53%** | NO |
| Logistic Regression | **0.00%** | NO |
| Decision Tree | **0.84%** | NO |

---

## 12. Key Findings

1. **StockOptionLevel** is the #1 Random Forest feature (12.41%) — employees with zero stock options leave far more often
2. **OverTime** has the strongest Pearson correlation (+0.246) — ~30.5% attrition rate vs ~10.4% without overtime
3. **JobSatisfaction** and **EnvironmentSatisfaction** are 2nd and 3rd most important features — directly actionable
4. **Sales Representatives** have the highest job-role attrition rate (~39.8%); Managers have the lowest (~5.1%)
5. **Single employees** leave ~2× more than married employees (25.5% vs 12.5%)
6. **Low Monthly Income** — employees earning below $4,787/month average leave significantly more often
7. **Frequent business travel** raises attrition from ~8% to ~24.9%
8. **Career stagnation** — long tenure but few years in current role — shows elevated attrition risk

---

## 13. Business Recommendations

### Immediate Actions
- Launch **stock option programmes** for Job Level 1–2 employees
- **Cap mandatory overtime**; introduce compensatory time-off
- **Flag employees** with JobSatisfaction or EnvironmentSatisfaction ≤ 2 for manager 1-on-1s
- Benchmark salaries for Job Level 1–2 against market; close gaps above 10%

### Medium-Term Actions
- Create **career ladders** for Sales Representatives and Laboratory Technicians
- Introduce **hybrid/remote work** for employees commuting more than 20 km
- Rotate or better compensate **frequent-travel assignments** equitably

### Strategic Actions
- **Integrate this model** into HR dashboards for monthly at-risk employee reports
- Build an **Employee Engagement Index** combining all four satisfaction scores
- Analyse **exit interview data** to continuously validate and improve model features

---

## 14. Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| Small dataset (1,470 rows) | Reduced generalisation | Collect more data; re-train periodically |
| Label encoding imposes ordinal order on nominal categories | May misrepresent relationships | Use One-Hot Encoding in future |
| SMOTE generates synthetic samples | Training distribution may differ from future real data | Monitor model drift in production |
| Static 80/20 split | Does not reflect temporal patterns | Use time-series cross-validation |
| Missing external factors | Job market, manager quality not captured | Enrich with survey/external data |
| Probabilistic output | Cannot guarantee individual-level accuracy | Combine with human HR judgment |

---

## 15. Conclusion

Three Machine Learning models were trained and evaluated on the IBM HR Analytics dataset:

- **Random Forest** achieved the best **ROC-AUC of 0.7348** and is deployed as the primary prediction model
- **Logistic Regression** achieved the best **Recall of 0.5106** — critical for catching true attrition cases
- **Decision Tree** provided an interpretable baseline with ROC-AUC of 0.6759

The top attrition drivers — **StockOptionLevel, JobSatisfaction, EnvironmentSatisfaction, MonthlyIncome, and OverTime** — are all directly actionable by HR and management teams.

Deploying this model in a real HR workflow, integrated with monthly reporting, can materially reduce voluntary turnover and its associated replacement cost, estimated at **50–200% of annual salary** per departing employee.

---

## 16. How to Run

### Prerequisites
- Python 3.9 or higher
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the same folder as `app.py`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Opens at: **http://localhost:8501**

### Navigation

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, attrition chart, dataset summary, pipeline steps |
| 🔍 EDA | 4 tabs: raw data, numeric histograms, categorical charts, correlations |
| 📈 Model Performance | Metrics table, confusion matrices, ROC curves, feature importances |
| 🔮 Predict Attrition | 30-field form → Yes/No + probability gauge + all-model votes |
| 💡 Insights | Findings, recommendations, limitations, conclusion |

---

## Project Files

```
ibm_project/
├── app.py                                    ← Complete Streamlit application
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    ← Dataset
├── requirements.txt                          ← Python dependencies
├── README.md                                 ← Project overview and run guide
├── PROJECT_REPORT.md                         ← This report
├── PROJECT_REPORT.html                       ← Interactive HTML version of report
└── screenshots/
    ├── screenshot_overview.png               ← UI screenshot: Overview page
    ├── screenshot_eda.png                    ← UI screenshot: EDA page
    ├── screenshot_performance.png            ← UI screenshot: Model Performance page
    ├── screenshot_predict.png                ← UI screenshot: Predict Attrition page
    └── screenshot_insights.png              ← UI screenshot: Insights page
```

---

*All results generated from the actual IBM HR Analytics dataset. No values are hardcoded or invented. Screenshots captured from the live running Streamlit application.*
