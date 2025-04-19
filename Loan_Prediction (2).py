import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

# UI Setup
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.title("💳 Prediksi Kelolosan Pinjaman")
st.markdown("Masukkan detail di bawah untuk memprediksi apakah **pinjaman akan disetujui** atau tidak.")

education_map, default_map = load_mappings()
model = load_model()

# Form Input
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'])
        education = st.selectbox("Pendidikan Terakhir", list(education_map.keys()))
        previous_default = st.selectbox("Pernah Gagal Bayar?", list(default_map.keys()))
        credit_score = st.slider("Skor Kredit (300 - 850)", min_value=300, max_value=850, value=650)

    with col2:
        income = st.number_input("Pendapatan Tahunan ($)", min_value=0)
        emp_exp = st.number_input("Lama Bekerja (Tahun)", min_value=0)
        age = st.number_input("Umur", min_value=18, max_value=100, value=35)
        cred_hist_len = st.number_input("Lama Riwayat Kredit (Tahun)", min_value=0)
        loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=0)
        loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0)
        loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

if submitted:
    # Manual encoding
    input_dict = {
        'person_income': income,
        'person_education': education_map[education],
        'loan_amnt': loan_amount,
        'loan_int_rate': loan_int_rate,
        'previous_loan_defaults_on_file': default_map[previous_default],
        'person_age': age,
        'person_emp_exp': emp_exp,
        'loan_percent_income': loan_amount / income if income > 0 else 0,
        'cb_person_cred_hist_length': cred_hist_len,
        'credit_score': credit_score
    }

    # One-hot encoding
    one_hot_features = {
        'person_gender_Male': 1 if gender == 'Male' else 0,
        'person_home_ownership_Own': 1 if home_ownership == 'Own' else 0,
        'person_home_ownership_Rent': 1 if home_ownership == 'Rent' else 0,
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
        'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == 'DEBTCONSOLIDATION' else 0,
    }

    input_dict.update(one_hot_features)
    input_df = pd.DataFrame([input_dict])

    # Reorder columns to match model input
    expected_features = model.get_booster().feature_names
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    # Output
    st.markdown("---")
    if prediction == 1:
        st.success("✅ Pinjaman kamu kemungkinan **DISETUJUI**! Selamat! 🎉")
    else:
        st.error("❌ Pinjaman kamu kemungkinan **DITOLAK**. Coba cek kembali detail pengajuanmu.")
