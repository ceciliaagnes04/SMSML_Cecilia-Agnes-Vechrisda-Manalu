# -*- coding: utf-8 -*-
"""dashboard_app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hB9gf2Ph01JeXVlEhui14ECRyRr0t0PB
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

st.title("Telco Customer Churn - Dashboard Eksperimen")
st.markdown("""
Dashboard ini menampilkan proses analisis, training, dan evaluasi model prediksi churn pelanggan Telco.
""")

st.header("1. Dataset")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
st.write("Contoh data:", df.head())
st.write("Ukuran data:", df.shape)

# Bersihkan TotalCharges kosong dan ubah ke float
df = df[df['TotalCharges'].astype(str).str.strip() != '']
df['TotalCharges'] = df['TotalCharges'].astype(float)

st.header("2. Eksplorasi Data")
col1, col2 = st.columns(2)
with col1:
    st.write("Distribusi Target (Churn):")
    st.bar_chart(df["Churn"].value_counts())
with col2:
    st.write("Distribusi Tenure:")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(df["tenure"], bins=20, color='skyblue')
    ax_hist.set_title("Distribusi Tenure")
    st.pyplot(fig_hist)

st.write("Heatmap Korelasi Numerik:")
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
corr = df[num_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig)

st.header("3. Preprocessing (Ringkasan)")
st.markdown("""
- Menghapus missing value pada TotalCharges
- Encoding fitur kategorikal dengan one-hot
- Scaling fitur numerik
- Split data train/test
""")

st.header("4. Hasil Preprocessing & Model")
train = pd.read_csv("Membangun_model/namadataset_preprocessing/train.csv")
test = pd.read_csv("Membangun_model/namadataset_preprocessing/test.csv")
X_test = test.drop("Churn", axis=1)
y_test = test["Churn"]
model = joblib.load("Membangun_model/model/churn_rf.pkl")

st.write("Contoh data preprocessing (train):", train.head())

st.header("5. Evaluasi Model")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
report = classification_report(y_test, y_pred, output_dict=True)
st.write("Classification Report:", pd.DataFrame(report).T)
st.write("ROC-AUC:", roc_auc_score(y_test, y_proba))

cm = confusion_matrix(y_test, y_pred)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax2)
st.pyplot(fig2)

st.header("6. Simulasi Prediksi Interaktif")
with st.form("predict_form"):
    st.write("Masukkan fitur customer:")
    input_data = {}
    for col in X_test.columns:
        if "Yes" in col or "No" in col or "Male" in col or "Female" in col:
            input_data[col] = st.selectbox(col, [0, 1])
        else:
            input_data[col] = st.number_input(col, value=float(X_test[col].mean()))
    submitted = st.form_submit_button("Prediksi Churn")
    if submitted:
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"Hasil prediksi: {'Churn' if pred==1 else 'Tidak Churn'} (Prob: {proba:.2f})")

st.header("7. Catatan Eksperimen")
st.markdown("""
- Model terbaik: RandomForestClassifier
- Akurasi: {:.2f}
- Precision: {:.2f}
- Recall: {:.2f}
- F1: {:.2f}
- ROC-AUC: {:.2f}
""".format(
    report["accuracy"],
    report["1"]["precision"],
    report["1"]["recall"],
    report["1"]["f1-score"],
    roc_auc_score(y_test, y_proba)
))