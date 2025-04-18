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

# --- UI Styling ---
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown(
    """
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        color: #5DADE2;
        margin-bottom: 20px;
    }
    .success-message {
        font-size: 24px;
        color: #27AE60;
        text-align: center;
    }
    .error-message {
        font-size: 24px;
        color: #E74C3C;
        text-align: center;
    }
    .form-container {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="header">💳 Prediksi Kelolosan Pinjaman</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Masukkan detail di bawah untuk memprediksi apakah pinjaman akan disetujui atau tidak.</p>', unsafe_allow_html=True)

# --- Form Input User ---
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'])
            education = st.selectbox("Pendidikan Terakhir", list(education_map.keys()))
            previous_default = st.selectbox("Pernah Gagal Bayar?", ['No', 'Yes'])

        with col2:
            income = st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000)
            loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000)
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1)
            loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Proses Prediksi ---
if submitted:
    # Manual encoding
    input_dict = {
        'person_income': income,
        'person_education': education_map[education],
        'loan_amnt': loan_amount,
        'loan_int_rate': loan_int_rate,
        'previous_loan_defaults_on_file': default_map[previous_default],
        # One-hot encoded fields
        'person_gender_Male': 1 if gender == 'Male' else 0,
        'person_home_ownership_Own': 1 if home_ownership == 'Own' else 0,
        'person_home_ownership_Rent': 1 if home_ownership == 'Rent' else 0,
        'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
    }
    
    input_df = pd.DataFrame([input_dict])
    expected_features = model.get_booster().feature_names
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    prediction = model.predict(input_df)[0]

    # --- Hasil ---
    st.markdown("---")
    if prediction == 1:
        st.markdown('<p class="success-message">✅ Pinjaman kamu kemungkinan <b>DISETUJUI</b>! Selamat! 🎉</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error-message">❌ Pinjaman kamu kemungkinan <b>DITOLAK</b>. Coba cek kembali detail pengajuanmu.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("💡 **Tips:** Pastikan semua informasi yang dimasukkan akurat untuk hasil prediksi yang lebih baik.")
