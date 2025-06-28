# ✅ Streamlit web app cho Cardio Risk, chạy 24/7 nếu deploy Render hoặc Streamlit Cloud

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Cardio Risk Classifier", layout="wide")
st.title("🚑 Cardio Health Risk Predictor")
st.write("Upload dữ liệu CSV (có cột 'cardio') và chạy mô hình Random Forest.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"]) 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 5 dòng đầu:")
    st.write(df.head())

    if 'id' in df.columns:
        X = df.drop(["id", "cardio"], axis=1)
    else:
        X = df.drop(["cardio"], axis=1)
    y = df["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.info("🔍 Đang chạy GridSearchCV...")

    param_grid = {
        "n_estimators": [100],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1]
    }

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)

    st.subheader("✅ Kết quả")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
    st.write(f"**F1:** {f1_score(y_test, y_pred):.2f}")
    st.write(f"**AUC:** {roc_auc_score(y_test, grid.best_estimator_.predict_proba(X_test)[:, 1]):.2f}")

    st.write("**Best Params:**", grid.best_params_)

    st.success("🎉 Huấn luyện xong! Bạn có thể deploy app này lên Render hoặc Streamlit Cloud để chạy 24/7.")

else:
    st.warning("⚠️ Hãy upload file CSV để bắt đầu!")
