import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Load dataset untuk ambil struktur fitur
@st.cache_data
def load_data():
    return pd.read_csv("Dataset_A_loan.csv")

# Preprocessing sederhana sesuai dengan notebook
# (kita sesuaikan nanti jika ada encoding lebih lanjut)
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df['person_gender'].replace('fe male', 'female', inplace=True)
    df['person_gender'].replace('Male', 'male', inplace=True)
    return df

# Load model dan mapping
@st.cache_resource
def load_model():
    with open("best_xgb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mappings():
    with open("education_map.pkl", "rb") as f:
        education_map = pickle.load(f)
    with open("default_map.pkl", "rb") as f:
        default_map = pickle.load(f)
    return education_map, default_map

# App UI
st.title("Loan Approval Prediction")
st.write("Masukkan detail peminjam untuk prediksi status pinjaman.")

data = load_data()

# Helper function untuk handle kolom tidak tersedia
def safe_selectbox(label, df, column_name, default_options):
    if column_name in df.columns:
        return st.selectbox(label, df[column_name].dropna().unique())
    else:
        return st.selectbox(label, default_options)

# Buat form input sesuai fitur penting
with st.form("prediction_form"):
    person_age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Pendapatan", value=50000)
    person_home_ownership = safe_selectbox("Status Tempat Tinggal", data, 'person_home_ownership', ['RENT', 'OWN', 'MORTGAGE'])
    person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0, max_value=50, value=5)
    loan_intent = safe_selectbox("Tujuan Pinjaman", data, 'loan_intent', ['EDUCATION', 'PERSONAL', 'VENTURE', 'MEDICAL'])
    loan_grade = safe_selectbox("Grade Pinjaman", data, 'loan_grade', ['A', 'B', 'C', 'D'])
    loan_amnt = st.number_input("Jumlah Pinjaman", value=10000)
    loan_int_rate = st.number_input("Suku Bunga (%)", value=10.5)
    loan_percent_income = st.number_input("Persentase dari Pendapatan", value=0.2)
    person_gender = st.selectbox("Jenis Kelamin", ['male', 'female'])
    submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'person_gender': person_gender
        }

        input_df = preprocess_input(input_data)

        model = load_model()
        prediction = model.predict(input_df)[0]
        st.success(f"Prediksi status pinjaman: {'Disetujui' if prediction == 1 else 'Ditolak'}")
