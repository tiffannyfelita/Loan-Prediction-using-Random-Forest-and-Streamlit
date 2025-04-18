import streamlit as st
import pickle
import pandas as pd

# --- Load Model & Mapping ---
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('education_map.pkl', 'rb') as f:
    education_map = pickle.load(f)

with open('default_map.pkl', 'rb') as f:
    default_map = pickle.load(f)

# --- UI Styling with Hirono Theme ---
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #F5F5F5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #34495E;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 18px;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 30px;
    }
    .form-container {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .success-message {
        font-size: 22px;
        color: #2ECC71;
        text-align: center;
        margin-top: 20px;
    }
    .error-message {
        font-size: 22px;
        color: #E74C3C;
        text-align: center;
        margin-top: 20px;
    }
    .submit-button {
        background-color: #3498DB;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .submit-button:hover {
        background-color: #2980B9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown('<p class="header">💳 Prediksi Kelolosan Pinjaman</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Masukkan detail untuk memprediksi apakah pinjaman akan disetujui atau tidak.</p>', unsafe_allow_html=True)

# --- Form Input User ---
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'], key="gender")
            home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'], key="home")
            education = st.selectbox("Pendidikan Terakhir", list(education_map.keys()), key="education")
            previous_default = st.selectbox("Pernah Gagal Bayar?", ['No', 'Yes'], key="default")

        with col2:
            income = st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000, key="income")
            loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000, key="loan_amount")
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1, key="loan_int_rate")
            loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], key="loan_intent")

        submitted = st.form_submit_button("🔍 Prediksi Sekarang", help="Klik untuk memproses prediksi")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Proses Prediksi ---
if submitted:
    # Manual encoding (termasuk semua kemungkinan one-hot)
    input_dict = {
        'person_income': income,
        'person_education': education_map[education],
        'loan_amnt': loan_amount,
        'loan_int_rate': loan_int_rate,
        'previous_loan_defaults_on_file': default_map[previous_default],

        # Gender
        'person_gender_Male': 1 if gender == 'Male' else 0,

        # Home ownership
        'person_home_ownership_Own': 1 if home_ownership == 'Own' else 0,
        'person_home_ownership_Rent': 1 if home_ownership == 'Rent' else 0,
        'person_home_ownership_Mortgage': 1 if home_ownership == 'Mortgage' else 0,

        # Loan intent
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
        'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == 'DEBTCONSOLIDATION' else 0,
    }

    # Konversi ke DataFrame
    input_df = pd.DataFrame([input_dict])

    # Pastikan urutan dan isi kolom cocok dengan model
    expected_features = model.get_booster().feature_names
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # --- Prediksi ---
    prediction = model.predict(input_df)[0]

    # --- Hasil ---
    st.markdown("---")
    if prediction == 1:
        st.markdown('<p class="success-message">✅ Pinjaman kamu kemungkinan <b>DISETUJUI</b>! Selamat! 🎉</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error-message">❌ Pinjaman kamu kemungkinan <b>DITOLAK</b>. Coba cek kembali detail pengajuanmu.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("💡 **Tips:** Pastikan semua informasi yang dimasukkan akurat untuk hasil prediksi yang lebih baik.")
