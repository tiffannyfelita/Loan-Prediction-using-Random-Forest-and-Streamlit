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

# --- Kategori Mapping untuk Fitur Categorical ---
gender_map = {'Male': 1, 'Female': 0}
home_map = {'Own': 1, 'Rent': 2, 'Mortgage': 3, 'Other': 4}
intent_map = {
    'EDUCATION': 0,
    'VENTURE': 1,
    'PERSONAL': 2,
    'MEDICAL': 3,
    'HOMEIMPROVEMENT': 4,
    'DEBTCONSOLIDATION': 5
}

# --- UI Styling ---
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="header">💳 Prediksi Kelolosan Pinjaman</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Masukkan data di bawah untuk melihat apakah pinjaman akan disetujui</p>', unsafe_allow_html=True)

# --- Form ---
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Usia", 18, 80, 30)
            gender = st.selectbox("Jenis Kelamin", list(gender_map.keys()))
            education = st.selectbox("Pendidikan", list(education_map.keys()))
            income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=50000, step=1000)
            emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 50, 5)
            credit_score = st.slider("Skor Kredit", 300, 850, 650)
            previous_default = st.selectbox("Pernah Gagal Bayar?", list(default_map.keys()))

        with col2:
            home = st.selectbox("Status Tempat Tinggal", list(home_map.keys()))
            loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=500, value=10000, step=500)
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
            loan_intent = st.selectbox("Tujuan Pinjaman", list(intent_map.keys()))
            cred_hist = st.slider("Lama Riwayat Kredit (tahun)", 0, 30, 8)

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Logic ---
if submitted:
    input_dict = {
        'person_age': age,
        'person_gender': gender_map[gender],
        'person_education': education_map[education],
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_map[home],
        'loan_amnt': loan_amount,
        'loan_intent': intent_map[loan_intent],
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_amount / income,
        'cb_person_cred_hist_length': cred_hist,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': default_map[previous_default],
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.markdown('<p class="success-message">✅ Pinjaman kamu kemungkinan <b>DISETUJUI</b>! 🎉</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error-message">❌ Pinjaman kamu kemungkinan <b>DITOLAK</b>. Periksa kembali data kamu.</p>', unsafe_allow_html=True)

    st.markdown("---")
