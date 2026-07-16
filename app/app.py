import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ===========================
# Title
# ===========================

st.title("💳 Credit Card Fraud Detection Dashboard")

st.markdown("""
This dashboard demonstrates a machine learning model built to detect fraudulent
credit card transactions using the **Kaggle Credit Card Fraud Detection Dataset**.

The project compares multiple machine learning algorithms and deploys the
best-performing model.
""")

st.divider()

# ===========================
# Dataset
# ===========================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Percentage", "0.172%")

st.info("""
The dataset is **highly imbalanced**, making Accuracy alone an unreliable evaluation metric.
Therefore Precision, Recall and F1-score were used while selecting the final model.
""")

st.divider()

# ===========================
# Final Model
# ===========================

st.header("🤖 Final Model")

st.write("""
After comparing different machine learning algorithms, the **Random Forest Classifier**
was selected as the final model because it achieved the best balance between
Precision and Recall.
""")

col1, col2 = st.columns(2)

col1.metric("Algorithm", "Random Forest")
col2.metric("Features", "30")

st.divider()

# ===========================
# Model Performance
# ===========================

st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.9%")
col2.metric("Precision", "97%")
col3.metric("Recall", "71%")
col4.metric("F1 Score", "82%")

st.success("""
Random Forest outperformed Logistic Regression on the fraud class,
making it the final deployed model.
""")

st.divider()

# ===========================
# Project Pipeline
# ===========================

st.header("⚙️ Project Pipeline")

st.markdown("""
1. Data Cleaning
2. Data Preprocessing
3. Train-Test Split
4. Feature Scaling
5. Logistic Regression
6. Model Evaluation
7. Random Forest Comparison
8. Model Serialization using Joblib
9. Streamlit Dashboard Deployment
""")

st.divider()

# ===========================
# Technologies
# ===========================

st.header("🛠️ Technologies Used")

st.write("""
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
""")

st.divider()

# ===========================
# About
# ===========================

st.header("📌 About This Project")

st.write("""
This project was developed as an end-to-end machine learning deployment project.

The dataset uses **anonymized PCA-transformed features (V1–V28)** to preserve
customer privacy. Therefore, instead of creating an unrealistic banking interface,
this application presents the model as an **analytical dashboard** that summarizes
the dataset, model selection process, and evaluation results.
""")