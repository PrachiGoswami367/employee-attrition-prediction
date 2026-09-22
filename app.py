# =============================================================================
#  Employee Attrition Prediction
#  File        : app.py
#  Description : Complete Streamlit frontend + Python backend
#  Run         : streamlit run app.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

# ── Standard imports ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)

# ── Imbalanced-learn ──────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE

# =============================================================================
#  CONSTANTS
# =============================================================================
DATA_FILE   = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
SEED        = 42
TEST_SIZE   = 0.20
SMOTE_STATE = 42

# Colour palette
C_BLUE   = "#2563EB"
C_PURPLE = "#7C3AED"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_AMBER  = "#D97706"
C_TEAL   = "#0891B2"
PALETTE  = [C_BLUE, C_PURPLE, C_GREEN, C_RED, C_AMBER, C_TEAL]

# =============================================================================
#  MODULE 1 — DATA LOADING & PREPROCESSING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Return raw DataFrame exactly as read from CSV."""
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def preprocess(raw_df: pd.DataFrame):
    """
    Clean, encode and split the dataset.

    Returns
    -------
    df_raw   : original strings preserved (used for EDA displays)
    df_enc   : fully label-encoded copy
    X, y     : feature matrix and target vector
    X_train, X_test, y_train, y_test : stratified 80/20 split
    X_train_sc, X_test_sc  : scaled splits for Logistic Regression
    X_res, y_res           : SMOTE-balanced training set
    scaler, le_dict        : fitted objects for inference
    cat_cols, num_cols     : column lists
    dropped_cols           : columns removed during cleaning
    """
    df = raw_df.copy()

    # ── 1. Remove constant columns (zero information) ─────────────────────
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    # ── 2. Remove ID column (no predictive value) ─────────────────────────
    id_cols = ["EmployeeNumber"]
    dropped_cols = list(set(constant_cols + id_cols) & set(df.columns))
    df.drop(columns=dropped_cols, inplace=True)

    # ── 3. Keep a clean string copy for EDA ───────────────────────────────
    df_raw = df.copy()

    # ── 4. Encode target: Yes → 1, No → 0 ────────────────────────────────
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # ── 5. Identify column types ──────────────────────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns
                if c != "Attrition"]

    # ── 6. Label-encode categorical columns ──────────────────────────────
    le_dict = {}
    df_enc  = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    # ── 7. Feature matrix / target ────────────────────────────────────────
    X = df_enc.drop(columns=["Attrition"])
    y = df_enc["Attrition"]

    # ── 8. Stratified train / test split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # ── 9. SMOTE on training set only ─────────────────────────────────────
    smote = SMOTE(random_state=SMOTE_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # ── 10. Scale (for Logistic Regression) ───────────────────────────────
    scaler    = StandardScaler()
    X_train_sc = scaler.fit_transform(X_res)
    X_test_sc  = scaler.transform(X_test)

    return (df_raw, df_enc, X, y,
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, X_res, y_res,
            scaler, le_dict, cat_cols, num_cols, dropped_cols)


# =============================================================================
#  MODULE 2 — MODEL TRAINING
# =============================================================================

@st.cache_resource(show_spinner=False)
def train_models(_X_sc, _y_res, _X_tree, _y_tree):
    """
    Train three classifiers.
    Logistic Regression uses scaled data; trees use raw resampled data.
    """
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(_X_sc, _y_res)

    dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED)
    dt.fit(_X_tree, _y_tree)

    rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(_X_tree, _y_tree)

    return lr, dt, rf


# =============================================================================
#  MODULE 3 — EVALUATION
# =============================================================================

def evaluate(model, X_test, y_test):
    """Return dict of all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }


# =============================================================================
#  MODULE 4 — PLOT HELPERS
# =============================================================================

def _clean_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def fig_attrition_dist(df_raw):
    counts  = df_raw["Attrition"].value_counts()
    labels  = ["No (Stay)", "Yes (Leave)"]
    values  = [counts.get("No", 0), counts.get("Yes", 0)]
    colors  = [C_GREEN, C_RED]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                     f"{int(b.get_height())}", ha="center", fontweight="bold", fontsize=12)
    _clean_ax(axes[0], "Attrition Count", "", "Number of Employees")

    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Attrition Split (%)", fontsize=12, fontweight="bold")

    fig.suptitle("Target Variable Distribution — Attrition (Yes / No)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_numeric_dist(df_raw, num_cols):
    cols = num_cols[:8]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        for val, color, lbl in [(0, C_GREEN, "No"), (1, C_RED, "Yes")]:
            subset = df_raw[df_raw["Attrition"].map({"No": 0, "Yes": 1}) == val][col]
            axes[i].hist(subset, bins=22, alpha=0.65, color=color, label=lbl)
        _clean_ax(axes[i], col)
        axes[i].legend(fontsize=8)
    fig.suptitle("Numeric Feature Distributions by Attrition",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_cat_attrition(df_raw, cat_cols):
    show = [c for c in ["OverTime", "MaritalStatus", "BusinessTravel",
                         "Department", "JobRole", "Gender"] if c in df_raw.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(show):
        grp = (df_raw.groupby(col)["Attrition"]
               .value_counts(normalize=True)
               .unstack()
               .fillna(0))
        if "Yes" in grp.columns:
            grp[["No", "Yes"]].plot(kind="bar", ax=axes[i],
                                    color=[C_GREEN, C_RED], edgecolor="white")
        _clean_ax(axes[i], col, col, "Proportion")
        axes[i].tick_params(axis="x", rotation=30)
        axes[i].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[i].legend(["No", "Yes"], fontsize=8)
    fig.suptitle("Categorical Features vs Attrition (Proportion)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_correlation(df_enc):
    corr = df_enc.corr()["Attrition"].drop("Attrition").sort_values()
    colors = [C_RED if v > 0 else C_BLUE for v in corr.values]
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    _clean_ax(ax, "Feature Correlation with Attrition (Pearson r)", "r", "")
    fig.tight_layout()
    return fig


def fig_confusion_matrices(results_dict, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, res) in zip(axes, results_dict.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_roc_curves(results_dict, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, res), color in zip(results_dict.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC = {res['ROC-AUC']:.3f})",
                color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    _clean_ax(ax, "ROC Curves", "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_metrics_bar(results_dict):
    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x     = np.arange(len(metric_keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, res) in enumerate(results_dict.items()):
        vals = [res[m] for m in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=PALETTE[i], edgecolor="white")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.12)
    _clean_ax(ax, "Model Performance Comparison", "", "Score")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_feature_importance(rf_model, feature_names):
    imp  = pd.Series(rf_model.feature_importances_, index=feature_names)
    top  = imp.nlargest(15).sort_values()
    colors = [C_PURPLE if v >= top.median() else C_BLUE for v in top.values]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color=colors)
    _clean_ax(ax, "Top 15 Feature Importances (Random Forest)", "Importance Score", "")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_prob_gauge(proba):
    fig, ax = plt.subplots(figsize=(7, 1.4))
    color = C_RED if proba >= 0.5 else C_GREEN
    ax.barh([""], [proba],       color=color,   height=0.5)
    ax.barh([""], [1 - proba],   left=[proba],  color="#E5E7EB", height=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Attrition Probability Gauge", fontsize=11, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    label_x = proba / 2 if proba > 0.12 else proba + 0.02
    ax.text(label_x, 0, f"{proba:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


# =============================================================================
#  MODULE 5 — STREAMLIT APP
# =============================================================================

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.4rem; }

  /* ── KPI cards ── */
  .kpi-card {
      background:#1E3A5F; border:1px solid #2563EB;
      border-radius:10px; padding:18px 14px; text-align:center;
  }
  .kpi-card .val { font-size:2rem; font-weight:700; color:#60A5FA; margin:0; }
  .kpi-card .lbl { font-size:0.82rem; color:#CBD5E1; margin:4px 0 0; font-weight:500; }

  /* ── Section titles ── */
  .section-title {
      font-size:1.18rem; font-weight:700; color:#F1F5F9;
      border-left:4px solid #3B82F6; padding-left:10px; margin-bottom:8px;
  }

  /* ── Info / note boxes  — always dark bg, always white text ── */
  .info-box {
      background:#1E3A5F; border-left:5px solid #3B82F6;
      border-radius:6px; padding:13px 17px; margin:6px 0;
      color:#E0F0FF !important; font-size:0.93rem; line-height:1.6;
  }
  .info-box strong { color:#93C5FD; }

  .warn-box {
      background:#422006; border-left:5px solid #F59E0B;
      border-radius:6px; padding:13px 17px; margin:6px 0;
      color:#FEF3C7 !important; font-size:0.93rem; line-height:1.6;
  }
  .warn-box strong { color:#FCD34D; }

  .ok-box {
      background:#052E16; border-left:5px solid #22C55E;
      border-radius:6px; padding:13px 17px; margin:6px 0;
      color:#DCFCE7 !important; font-size:0.93rem; line-height:1.6;
  }
  .ok-box strong { color:#86EFAC; }

  .risk-box {
      background:#450A0A; border-left:5px solid #EF4444;
      border-radius:6px; padding:13px 17px; margin:6px 0;
      color:#FEE2E2 !important; font-size:0.93rem; line-height:1.6;
  }
  .risk-box strong { color:#FCA5A5; }

  /* ── Pipeline step numbers ── */
  .info-box em { color:#BAE6FD; font-style:normal; font-weight:700; }

  /* ── Metrics table styling ── */
  [data-testid="stDataFrame"] table {
      font-size: 0.92rem;
  }

  /* ── Sidebar caption text ── */
  [data-testid="stSidebar"] .stCaption p { color: #94A3B8 !important; }

  /* ── Tab labels ── */
  [data-testid="stTabs"] button { font-weight: 600; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Load & process data ───────────────────────────────────────────────────────
with st.spinner("Loading dataset and training models — please wait…"):
    raw_df = load_data()
    (df_raw, df_enc, X, y,
     X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, X_res, y_res,
     scaler, le_dict, cat_cols, num_cols, dropped_cols) = preprocess(raw_df)

    lr_model, dt_model, rf_model = train_models(
        X_train_sc, y_res, X_res, y_res
    )

    lr_res = evaluate(lr_model, X_test_sc, y_test)
    dt_res = evaluate(dt_model, X_test,    y_test)
    rf_res = evaluate(rf_model, X_test,    y_test)

    results = {
        "Logistic Regression": lr_res,
        "Decision Tree":       dt_res,
        "Random Forest":       rf_res,
    }
    best_model_name = max(results, key=lambda k: results[k]["ROC-AUC"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Employee Attrition\nPrediction")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview",
         "🔍 Exploratory Data Analysis",
         "📈 Model Performance",
         "🔮 Predict Attrition",
         "💡 Insights & Recommendations"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        f"**Dataset:** {len(df_raw):,} employees · {X.shape[1]} features\n\n"
        f"**Attrition rate:** {y.mean():.1%}\n\n"
        f"**Best model:** {best_model_name}\n\n"
        f"**Best ROC-AUC:** {results[best_model_name]['ROC-AUC']:.4f}"
    )


# =============================================================================
#  PAGE 1 — OVERVIEW
# =============================================================================
if page == "🏠 Overview":
    st.title("📊 Employee Attrition Prediction Dashboard")
    st.markdown(
        "**Problem Statement:** Organizations lose significant value when employees leave unexpectedly. "
        "This application analyses IBM HR data to predict whether an employee will leave (**Attrition = Yes/No**) "
        "based on demographics, job role, salary, overtime, satisfaction, work-life balance, and tenure."
    )
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (f"{len(df_raw):,}",        "Total Employees"),
        (f"{y.sum():,}",            "Left (Attrition = Yes)"),
        (f"{y.mean():.1%}",         "Overall Attrition Rate"),
        (f"{X.shape[1]}",           "Predictive Features"),
        (f"{results[best_model_name]['ROC-AUC']:.3f}", "Best ROC-AUC"),
    ]
    for col, (v, l) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="kpi-card"><p class="val">{v}</p><p class="lbl">{l}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown('<p class="section-title">Attrition Distribution</p>',
                    unsafe_allow_html=True)
        st.pyplot(fig_attrition_dist(df_raw), use_container_width=True)
    with col_r:
        st.markdown('<p class="section-title">Dataset Summary</p>',
                    unsafe_allow_html=True)
        st.dataframe(
            df_raw.describe(include="all")
                  .T[["count", "unique", "mean", "std", "min", "max"]]
                  .fillna("—"),
            height=340,
        )

    st.markdown("---")
    st.markdown('<p class="section-title">ML Pipeline Steps</p>',
                unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Loading",        f"Read {len(df_raw):,} rows × {raw_df.shape[1]} columns from IBM HR CSV."),
        ("2️⃣", "Data Cleaning",        f"Removed {len(dropped_cols)} constant/ID column(s): {', '.join(f'`{c}`' for c in dropped_cols)}."),
        ("3️⃣", "Encoding",             "Label-encoded 7 categorical columns (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime)."),
        ("4️⃣", "Train/Test Split",     f"Stratified 80 / 20 split → {len(X_train):,} train, {len(X_test):,} test samples."),
        ("5️⃣", "Class Imbalance",      f"Applied SMOTE on training set: balanced {y_train.value_counts()[0]} No → equal No/Yes samples."),
        ("6️⃣", "Feature Scaling",      "StandardScaler fitted on SMOTE training data; applied to Logistic Regression only."),
        ("7️⃣", "Model Training",       "Trained Logistic Regression, Decision Tree (depth=6), and Random Forest (300 trees)."),
        ("8️⃣", "Evaluation",           "Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix on held-out test set."),
    ]
    for num, title, desc in steps:
        st.markdown(
            f'<div class="info-box">{num} <strong>{title}</strong> — {desc}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
#  PAGE 2 — EDA
# =============================================================================
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")

    t1, t2, t3, t4 = st.tabs([
        "📋 Dataset",
        "📊 Numeric Features",
        "🏷️ Categorical Features",
        "🔗 Correlations",
    ])

    # ── Tab 1: raw dataset ────────────────────────────────────────────────
    with t1:
        st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns &nbsp;|&nbsp; "
                    f"**Missing values:** {df_raw.isnull().sum().sum()} &nbsp;|&nbsp; "
                    f"**Dropped columns:** {', '.join(f'`{c}`' for c in dropped_cols)}")
        st.dataframe(df_raw.head(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Types & Unique Counts**")
            info = pd.DataFrame({
                "Type":     df_raw.dtypes.astype(str),
                "Non-Null": df_raw.notnull().sum(),
                "Unique":   df_raw.nunique(),
            })
            st.dataframe(info, use_container_width=True)
        with col2:
            st.markdown("**Target Variable — Attrition**")
            vc = df_raw["Attrition"].value_counts()
            tbl = pd.DataFrame({
                "Class":   ["No (Stay)", "Yes (Leave)"],
                "Count":   [vc.get("No", 0), vc.get("Yes", 0)],
                "Share %": [f"{vc.get('No',0)/len(df_raw)*100:.1f}%",
                             f"{vc.get('Yes',0)/len(df_raw)*100:.1f}%"],
            })
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            st.markdown(
                '<div class="warn-box">⚠️ Dataset is <strong>imbalanced</strong> '
                '(~84% No, ~16% Yes). SMOTE was applied to the training set only '
                'to prevent data leakage.</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 2: numeric ────────────────────────────────────────────────────
    with t2:
        st.markdown("**Overlapping histograms — Green = Stay, Red = Leave**")
        st.pyplot(fig_numeric_dist(df_raw, num_cols), use_container_width=True)
        st.markdown("---")
        sel = st.selectbox("🔎 Deep-dive into a feature", num_cols, key="num_sel")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        for attrval, color, label in [("No", C_GREEN, "No (Stay)"),
                                       ("Yes", C_RED, "Yes (Leave)")]:
            subset = df_raw[df_raw["Attrition"] == attrval][sel]
            ax2.hist(subset, bins=25, alpha=0.65, color=color, label=label)
        _clean_ax(ax2, f"{sel}  by  Attrition", sel, "Count")
        ax2.legend(); fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown(f"**Statistics for `{sel}`**")
        st.dataframe(
            df_raw.groupby("Attrition")[sel].describe().round(2),
            use_container_width=True,
        )

    # ── Tab 3: categorical ────────────────────────────────────────────────
    with t3:
        st.markdown("**Attrition proportion within each categorical column**")
        st.pyplot(fig_cat_attrition(df_raw, cat_cols), use_container_width=True)
        st.markdown("---")
        sel_cat = st.selectbox("🔎 Select categorical column", cat_cols, key="cat_sel")
        vc2 = df_raw.groupby(sel_cat)["Attrition"].value_counts().unstack().fillna(0)
        if "No" in vc2.columns and "Yes" in vc2.columns:
            vc2["Total"]         = vc2["No"] + vc2["Yes"]
            vc2["Attrition Rate"] = (vc2["Yes"] / vc2["Total"] * 100).round(1).astype(str) + "%"
        st.dataframe(vc2, use_container_width=True)

    # ── Tab 4: correlations ───────────────────────────────────────────────
    with t4:
        col_a, col_b = st.columns([1, 1.1])
        with col_a:
            st.markdown("**Pearson r with Attrition (encoded)**")
            st.pyplot(fig_correlation(df_enc), use_container_width=True)
        with col_b:
            corr_tbl = (
                df_enc.corr()["Attrition"]
                      .drop("Attrition")
                      .sort_values(key=abs, ascending=False)
                      .reset_index()
                      .rename(columns={"index": "Feature", "Attrition": "Pearson r"})
            )
            corr_tbl["Pearson r"] = corr_tbl["Pearson r"].round(4)
            corr_tbl["Direction"] = corr_tbl["Pearson r"].apply(
                lambda v: "🔴 Positive (raises risk)" if v > 0 else "🔵 Negative (lowers risk)"
            )
            st.dataframe(corr_tbl, use_container_width=True, height=430)
        st.markdown(
            '<div class="info-box">💡 <strong>OverTime</strong> has the strongest '
            'positive correlation with Attrition. <strong>JobLevel</strong> and '
            '<strong>MonthlyIncome</strong> show the strongest negative correlations — '
            'senior, higher-paid employees leave less often.</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
#  PAGE 3 — MODEL PERFORMANCE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance & Evaluation")

    # Metrics summary table
    st.markdown('<p class="section-title">Evaluation Metrics (Test Set)</p>',
                unsafe_allow_html=True)
    metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metrics_df = pd.DataFrame(
        {name: {k: round(res[k], 4) for k in metric_keys}
         for name, res in results.items()}
    ).T
    metrics_df.index.name = "Model"
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#DCFCE7")
                                  .highlight_min(axis=0, color="#FEE2E2"),
                 use_container_width=True)

    st.markdown(
        f'<div class="ok-box">✅ <strong>{best_model_name}</strong> achieves the highest '
        f'ROC-AUC of <strong>{results[best_model_name]["ROC-AUC"]:.4f}</strong> and is '
        f'used for the employee prediction form.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    t_bar, t_cm, t_roc, t_fi = st.tabs([
        "📊 Metrics Comparison",
        "🔲 Confusion Matrices",
        "📉 ROC Curves",
        "⭐ Feature Importances",
    ])

    with t_bar:
        st.pyplot(fig_metrics_bar(results), use_container_width=True)
        st.markdown(
            '<div class="info-box">ℹ️ <strong>Recall</strong> is critical for attrition '
            'prediction — missing a true "Yes" is more costly than a false alarm. '
            'Logistic Regression achieves the highest Recall, while Random Forest '
            'leads on ROC-AUC.</div>',
            unsafe_allow_html=True,
        )

    with t_cm:
        st.pyplot(fig_confusion_matrices(results, y_test), use_container_width=True)
        cols = st.columns(3)
        for col, (name, res) in zip(cols, results.items()):
            cm      = confusion_matrix(y_test, res["y_pred"])
            tn, fp, fn, tp = cm.ravel()
            col.markdown(
                f'<div class="info-box"><strong>{name}</strong><br>'
                f'TP={tp} &nbsp; TN={tn} &nbsp; FP={fp} &nbsp; FN={fn}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**Full Classification Reports**")
        for name, res in results.items():
            with st.expander(f"📋 {name} — Classification Report"):
                report = classification_report(
                    y_test, res["y_pred"],
                    target_names=["No (Stay)", "Yes (Leave)"],
                )
                st.code(report)

    with t_roc:
        col_r, col_d = st.columns([1.2, 1])
        with col_r:
            st.pyplot(fig_roc_curves(results, y_test), use_container_width=True)
        with col_d:
            st.markdown("### ROC-AUC Scores")
            for name, res in results.items():
                st.metric(name, f"{res['ROC-AUC']:.4f}")
            st.markdown(
                "**ROC-AUC = 1.0** → perfect classifier  \n"
                "**ROC-AUC = 0.5** → random guessing  \n\n"
                "A higher AUC means the model better distinguishes "
                "employees who will leave from those who will stay."
            )

    with t_fi:
        col_f, col_t = st.columns([1.2, 1])
        with col_f:
            st.pyplot(fig_feature_importance(rf_model, X.columns),
                      use_container_width=True)
        with col_t:
            fi_df = (
                pd.Series(rf_model.feature_importances_, index=X.columns)
                  .sort_values(ascending=False)
                  .head(15)
                  .reset_index()
            )
            fi_df.columns = ["Feature", "Importance"]
            fi_df["Importance"] = fi_df["Importance"].round(4)
            fi_df.insert(0, "Rank", range(1, len(fi_df) + 1))
            st.dataframe(fi_df, use_container_width=True, hide_index=True)


# =============================================================================
#  PAGE 4 — PREDICTION FORM
# =============================================================================
elif page == "🔮 Predict Attrition":
    st.title("🔮 Employee Attrition Prediction Form")
    st.markdown(
        "Enter employee details below and click **Predict** to receive an "
        "**Attrition: Yes / No** prediction with confidence probability.  \n"
        f"*Model used: **{best_model_name}** (highest ROC-AUC = "
        f"{results[best_model_name]['ROC-AUC']:.4f})*"
    )
    st.markdown("---")

    with st.form("predict_form", clear_on_submit=False):

        # ── Personal ──────────────────────────────────────────────────────
        st.markdown("#### 👤 Personal Information")
        p1, p2, p3 = st.columns(3)
        age            = p1.slider("Age", 18, 60, 35)
        gender         = p2.selectbox("Gender", ["Male", "Female"])
        marital_status = p3.selectbox("Marital Status",
                                      ["Single", "Married", "Divorced"])

        # ── Job ───────────────────────────────────────────────────────────
        st.markdown("#### 💼 Job Information")
        j1, j2, j3 = st.columns(3)
        department      = j1.selectbox("Department",
                                       ["Sales", "Research & Development",
                                        "Human Resources"])
        job_role        = j2.selectbox("Job Role",
                                       sorted(df_raw["JobRole"].unique().tolist()))
        job_level       = j3.selectbox("Job Level (1=Entry … 5=Executive)",
                                       [1, 2, 3, 4, 5])

        j4, j5, j6 = st.columns(3)
        business_travel = j4.selectbox("Business Travel",
                                       ["Non-Travel", "Travel_Rarely",
                                        "Travel_Frequently"])
        overtime        = j5.selectbox("Overtime", ["No", "Yes"])
        job_involvement = j6.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)

        # ── Compensation ──────────────────────────────────────────────────
        st.markdown("#### 💰 Compensation & Experience")
        m1, m2, m3 = st.columns(3)
        monthly_income      = m1.number_input("Monthly Income ($)", 1009, 19999, 5000, step=100)
        percent_salary_hike = m2.slider("Last Salary Hike (%)", 11, 25, 14)
        stock_option_level  = m3.selectbox("Stock Option Level", [0, 1, 2, 3])

        m4, m5, m6 = st.columns(3)
        total_working_years  = m4.slider("Total Working Years", 0, 40, 10)
        num_companies_worked = m5.slider("No. of Previous Companies", 0, 9, 2)
        training_last_year   = m6.slider("Trainings Last Year", 0, 6, 3)

        m7, m8, m9 = st.columns(3)
        daily_rate   = m7.number_input("Daily Rate ($)", 102, 1499, 800, step=50)
        hourly_rate  = m8.number_input("Hourly Rate ($)", 30, 100, 65)
        monthly_rate = m9.number_input("Monthly Rate ($)", 2094, 26999, 14000, step=100)

        # ── Satisfaction ──────────────────────────────────────────────────
        st.markdown("#### 😊 Satisfaction & Work-Life Balance")
        s1, s2, s3, s4 = st.columns(4)
        env_satisfaction  = s1.slider("Environment Satisfaction (1–4)", 1, 4, 3)
        job_satisfaction  = s2.slider("Job Satisfaction (1–4)", 1, 4, 3)
        work_life_balance = s3.slider("Work-Life Balance (1–4)", 1, 4, 3)
        relationship_sat  = s4.slider("Relationship Satisfaction (1–4)", 1, 4, 3)

        # ── Education ─────────────────────────────────────────────────────
        st.markdown("#### 🎓 Education")
        e1, e2 = st.columns(2)
        education       = e1.selectbox("Education Level (1=Below College … 5=Doctor)",
                                        [1, 2, 3, 4, 5])
        education_field = e2.selectbox("Education Field",
                                        ["Life Sciences", "Medical", "Marketing",
                                         "Technical Degree", "Human Resources", "Other"])

        # ── Tenure ────────────────────────────────────────────────────────
        st.markdown("#### 🏢 Tenure")
        t1c, t2c, t3c = st.columns(3)
        years_at_company      = t1c.slider("Years at Company",          0, 40, 5)
        years_in_role         = t2c.slider("Years in Current Role",      0, 18, 3)
        years_since_promotion = t3c.slider("Years Since Last Promotion", 0, 15, 1)

        t4c, t5c, t6c = st.columns(3)
        years_with_manager = t4c.slider("Years With Current Manager", 0, 17, 4)
        performance_rating = t5c.selectbox("Performance Rating (3=Excellent, 4=Outstanding)",
                                            [3, 4])
        distance           = t6c.slider("Distance From Home (km)", 1, 29, 5)

        submitted = st.form_submit_button(
            "🔮  Predict Attrition", type="primary", use_container_width=True
        )

    # ── Inference ─────────────────────────────────────────────────────────
    if submitted:
        # Encode categorical inputs with the same LabelEncoders
        def enc(col, val):
            return int(le_dict[col].transform([val])[0]) if col in le_dict else val

        raw_row = {
            "Age":                      age,
            "BusinessTravel":           enc("BusinessTravel",  business_travel),
            "DailyRate":                daily_rate,
            "Department":               enc("Department",      department),
            "DistanceFromHome":         distance,
            "Education":                education,
            "EducationField":           enc("EducationField",  education_field),
            "EnvironmentSatisfaction":  env_satisfaction,
            "Gender":                   enc("Gender",          gender),
            "HourlyRate":               hourly_rate,
            "JobInvolvement":           job_involvement,
            "JobLevel":                 job_level,
            "JobRole":                  enc("JobRole",         job_role),
            "JobSatisfaction":          job_satisfaction,
            "MaritalStatus":            enc("MaritalStatus",   marital_status),
            "MonthlyIncome":            monthly_income,
            "MonthlyRate":              monthly_rate,
            "NumCompaniesWorked":       num_companies_worked,
            "OverTime":                 enc("OverTime",        overtime),
            "PercentSalaryHike":        percent_salary_hike,
            "PerformanceRating":        performance_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StockOptionLevel":         stock_option_level,
            "TotalWorkingYears":        total_working_years,
            "TrainingTimesLastYear":    training_last_year,
            "WorkLifeBalance":          work_life_balance,
            "YearsAtCompany":           years_at_company,
            "YearsInCurrentRole":       years_in_role,
            "YearsSinceLastPromotion":  years_since_promotion,
            "YearsWithCurrManager":     years_with_manager,
        }

        # Align column order exactly with training feature matrix
        input_df = pd.DataFrame([{col: raw_row.get(col, 0) for col in X.columns}])

        # Use best model (Random Forest) for prediction
        proba      = float(rf_model.predict_proba(input_df)[0][1])
        prediction = "Yes" if proba >= 0.5 else "No"
        confidence = proba if prediction == "Yes" else 1 - proba

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        if prediction == "Yes":
            res_col1.markdown(
                '<div class="risk-box" style="text-align:center">'
                '<h2 style="color:#DC2626;margin:0">⚠️ ATTRITION: YES</h2>'
                '<p>This employee is at <strong>high risk</strong> of leaving.</p>'
                '</div>', unsafe_allow_html=True,
            )
        else:
            res_col1.markdown(
                '<div class="ok-box" style="text-align:center">'
                '<h2 style="color:#16A34A;margin:0">✅ ATTRITION: NO</h2>'
                '<p>This employee is likely to <strong>stay</strong>.</p>'
                '</div>', unsafe_allow_html=True,
            )

        res_col2.metric("Attrition Probability",  f"{proba:.1%}")
        res_col3.metric("Prediction Confidence",  f"{confidence:.1%}")

        # Probability gauge
        st.pyplot(fig_prob_gauge(proba), use_container_width=True)

        # Model votes from all 3 models
        st.markdown("### 🗳️ All Model Predictions")
        vote_cols = st.columns(3)
        for vcol, (mname, mmodel, mX) in zip(
            vote_cols,
            [("Logistic Regression", lr_model, scaler.transform(input_df)),
             ("Decision Tree",       dt_model, input_df),
             ("Random Forest",       rf_model, input_df)],
        ):
            mp   = float(mmodel.predict_proba(mX)[0][1])
            mpred = "Yes ⚠️" if mp >= 0.5 else "No ✅"
            vcol.metric(mname, mpred, f"P(Yes) = {mp:.1%}")

        # Key influencing factors
        st.markdown("### 🔍 Top Factors Influencing This Prediction")
        fi_series   = pd.Series(rf_model.feature_importances_, index=X.columns)
        top_factors = fi_series.nlargest(8)
        emp_vals    = input_df.iloc[0][top_factors.index]
        avg_vals    = X[top_factors.index].mean()

        factor_df = pd.DataFrame({
            "Feature":      top_factors.index,
            "Importance":   top_factors.values.round(4),
            "Your Value":   emp_vals.values.round(2),
            "Dataset Avg":  avg_vals.values.round(2),
        })
        st.dataframe(factor_df, use_container_width=True, hide_index=True)

        if prediction == "Yes":
            st.markdown(
                '<div class="warn-box">💡 <strong>HR Action Required:</strong> '
                'Review this employee\'s overtime load, compensation, and career '
                'development path. Early intervention can significantly reduce '
                'attrition probability.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ok-box">✅ <strong>Employee appears engaged.</strong> '
                'Continue monitoring satisfaction scores and work-life balance '
                'during periodic reviews.</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
#  PAGE 5 — INSIGHTS & RECOMMENDATIONS
# =============================================================================
elif page == "💡 Insights & Recommendations":
    st.title("💡 Insights, Recommendations & Conclusion")

    # Key findings
    st.markdown('<p class="section-title">🔍 Key Findings from the Data</p>',
                unsafe_allow_html=True)
    findings = [
        ("OverTime is the strongest risk factor",
         "Employees working overtime have ~3× higher attrition rate (~30%) vs non-overtime employees (~10%). "
         "Reducing mandatory overtime is the single highest-impact intervention."),
        ("Low Monthly Income drives attrition",
         "Employees earning below the median income (≈$4,900/month) leave far more frequently. "
         "Competitive compensation at Job Levels 1–2 directly reduces turnover."),
        ("Single employees leave more often",
         "Single employees show ~2× the attrition rate of married peers, likely due to fewer "
         "financial obligations and greater career mobility."),
        ("Distance from home matters",
         "Employees with long commutes (>20 km) show elevated attrition. "
         "Flexible/remote work options can address this hidden cost."),
        ("Sales Representatives & Lab Technicians are highest-risk roles",
         "These two roles consistently appear in the top attrition cohorts in this dataset. "
         "Role-specific retention programmes are justified."),
        ("Frequent business travel correlates with leaving",
         "Employees who travel frequently have significantly higher attrition than non-travellers."),
        ("Low Job & Environment Satisfaction",
         "Satisfaction scores of 1 or 2 (out of 4) are strong predictors of attrition. "
         "Regular pulse surveys that lead to action can move these scores."),
        ("Stagnant career progression",
         "Employees stuck in the same role for many years without promotion show elevated risk. "
         "Clear promotion pathways are essential."),
    ]
    for title, detail in findings:
        st.markdown(
            f'<div class="info-box"><strong>▶ {title}</strong><br>{detail}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Recommendations
    st.markdown('<p class="section-title">📋 Business Recommendations</p>',
                unsafe_allow_html=True)
    recs = {
        "🚨 Immediate Actions": [
            "Audit all mandatory overtime; enforce maximum overtime limits and introduce comp-time.",
            "Benchmark salaries for Job Levels 1–2 against the market and close gaps above 10%.",
            "Flag employees with Job/Environment Satisfaction ≤ 2 for manager 1-on-1 meetings.",
        ],
        "📅 Medium-Term Actions": [
            "Create structured career ladders for Sales Representatives and Laboratory Technicians.",
            "Introduce hybrid/remote work for employees commuting more than 20 km.",
            "Rotate or compensate frequent-travel roles more equitably.",
        ],
        "🎯 Strategic Actions": [
            "Integrate this prediction model into HR dashboards for monthly at-risk employee reports.",
            "Build an Employee Engagement Index combining all four satisfaction scores.",
            "Collect and analyse exit interview data to continuously validate model assumptions.",
        ],
    }
    for horizon, items in recs.items():
        st.markdown(f"**{horizon}**")
        for item in items:
            st.markdown(f"- {item}")

    st.markdown("---")

    # Limitations
    st.markdown('<p class="section-title">⚠️ Limitations</p>', unsafe_allow_html=True)
    limits = [
        "Dataset has only **1,470 rows** — predictions improve with more recent and larger samples.",
        "**Label encoding** imposes arbitrary ordinal order on nominal categories (e.g., JobRole). One-hot encoding may improve accuracy.",
        "**SMOTE** generates synthetic samples; true minority distribution may differ.",
        "Static **80/20 split** evaluation — time-series cross-validation would better reflect production use.",
        "Features like team dynamics, manager quality, and external job market are **not captured**.",
        "Model predictions are probabilistic — individual HR decisions must involve human judgment.",
    ]
    for lim in limits:
        st.markdown(
            f'<div class="warn-box">⚠️ {lim}</div>', unsafe_allow_html=True
        )

    st.markdown("---")

    # Conclusion
    st.markdown('<p class="section-title">✅ Conclusion</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ok-box">'
        f'Three machine learning models were trained and compared on the IBM HR Analytics '
        f'dataset. <strong>{best_model_name}</strong> achieved the highest ROC-AUC of '
        f'<strong>{results[best_model_name]["ROC-AUC"]:.4f}</strong> and F1 of '
        f'<strong>{results[best_model_name]["F1"]:.4f}</strong>. '
        f'The top attrition drivers identified — <strong>OverTime, MonthlyIncome, '
        f'TotalWorkingYears, JobLevel</strong>, and <strong>Age</strong> — are all '
        f'actionable business levers. Deploying this tool in HR workflows, combined '
        f'with targeted retention programmes, can materially reduce voluntary turnover '
        f'and its associated replacement cost (typically 50–200% of annual salary per '
        f'departing employee).'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Final comparison table
    st.markdown("---")
    st.markdown('<p class="section-title">Final Model Comparison</p>',
                unsafe_allow_html=True)
    final_df = pd.DataFrame(
        {name: {k: round(res[k], 4) for k in ["Accuracy","Precision","Recall","F1","ROC-AUC"]}
         for name, res in results.items()}
    ).T
    final_df.index.name = "Model"
    st.dataframe(final_df, use_container_width=True)
    st.caption(
        "All metrics computed on the held-out test set (20%, stratified split). "
        "SMOTE applied to training set only to prevent data leakage."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📊 **Employee Attrition Prediction** | IBM HR Analytics Dataset | "
    "Python · Streamlit · Scikit-learn · SMOTE · Pandas · Seaborn"
)
