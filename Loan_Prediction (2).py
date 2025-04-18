import streamlit as st
import pandas as pd
import pickle

# === Load Model dan Mapping ===
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("education_map.pkl", "rb") as f:
    education_map = pickle.load(f)

with open("default_map.pkl", "rb") as f:
    default_map = pickle.load(f)

# === UI Styling ===
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown("""
<style>
.form-container {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
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

st.markdown('<h2 style="text-align:center;">💳 Prediksi Kelolosan Pinjaman</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray;">Masukkan semua data berikut secara lengkap untuk hasil akurat</p>', unsafe_allow_html=True)

# === Form Input Slide Style ===
with st.form("loan_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    st.slider("Umur Pemohon", 18, 100, 30, key="person_age")
    st.slider("Lama Bekerja (tahun)", 0, 40, 5, key="person_emp_exp")
    st.slider("Panjang Riwayat Kredit (tahun)", 0, 30, 3, key="cb_person_cred_hist_length")
    st.slider("Skor Kredit (300-850)", 300, 850, 600, key="credit_score")

    st.selectbox("Jenis Kelamin", ['Male', 'Female'], key="gender")
    st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'], key="home")
    st.selectbox("Pendidikan Terakhir", list(education_map.keys()), key="education")
    st.selectbox("Pernah Gagal Bayar?", ['No', 'Yes'], key="default")

    st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000, key="income")
    st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000, key="loan_amount")
    st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1, key="loan_int_rate")

    st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], key="loan_intent")

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    st.markdown('</div>', unsafe_allow_html=True)

# === Proses Prediksi ===
if submitted:
    # Hitung fitur tambahan
    loan_percent_income = st.session_state.loan_amount / (st.session_state.income + 1e-5)

    input_dict = {
        'person_age': st.session_state.person_age,
        'person_emp_exp': st.session_state.person_emp_exp,
        'cb_person_cred_hist_length': st.session_state.cb_person_cred_hist_length,
        'credit_score': st.session_state.credit_score,
        'person_income': st.session_state.income,
        'person_education': education_map[st.session_state.education],
        'loan_amnt': st.session_state.loan_amount,
        'loan_int_rate': st.session_state.loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'previous_loan_defaults_on_file': default_map[st.session_state.default],

        # One-hot encoding
        'person_gender_Male': 1 if st.session_state.gender == 'Male' else 0,
        'person_home_ownership_Own': 1 if st.session_state.home == 'Own' else 0,
        'person_home_ownership_Rent': 1 if st.session_state.home == 'Rent' else 0,
        'loan_intent_EDUCATION': 1 if st.session_state.loan_intent == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if st.session_state.loan_intent == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if st.session_state.loan_intent == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if st.session_state.loan_intent == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if st.session_state.loan_intent == 'VENTURE' else 0,
    }

    # Buat DataFrame & urutkan fitur
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.markdown('<p class="success-message">✅ Pinjaman kamu kemungkinan <b>DISETUJUI</b>! Selamat! 🎉</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="error-message">❌ Pinjaman kamu kemungkinan <b>DITOLAK</b>. Coba cek kembali detail pengajuanmu.</p>', unsafe_allow_html=True)
