import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

class LoanPredictor:
    def __init__(self):
        self.model = self.load_model()
        self.education_map, self.default_map = self.load_mappings()

    def load_model(self):
        with open("best_xgb_model.pkl", "rb") as f:
            return pickle.load(f)

    def load_mappings(self):
        with open("education_map.pkl", "rb") as f:
            education_map = pickle.load(f)
        with open("default_map.pkl", "rb") as f:
            default_map = pickle.load(f)
        return education_map, default_map

    def predict(self, input_features):
        input_df = pd.DataFrame([input_features])
        expected_features = self.model.get_booster().feature_names
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        probability = self.model.predict_proba(input_df)[0][1]
        prediction = int(probability > 0.35)
        return prediction, probability

# Streamlit App
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.title("💳 Aplikasi Prediksi Kelolosan Pinjaman")

predictor = LoanPredictor()

menu = ["🏠 Beranda", "💸 Simulasi Pinjaman", "📋 Formulir Prediksi"]
choice = st.sidebar.selectbox("Navigasi", menu)

if choice == "🏠 Beranda":
    st.subheader("Selamat datang di Aplikasi Prediksi Pinjaman! 💸")
    st.write("Aplikasi ini membantu Anda melakukan simulasi pinjaman dan memprediksi kemungkinan pinjaman Anda akan disetujui atau tidak.")
    st.write("UTS Model Deployment 2025 - Tiffanny Felita")
    st.markdown("---")
    st.info("Pilih menu di sebelah kiri untuk mulai melakukan simulasi atau prediksi.")

elif choice == "💸 Simulasi Pinjaman":
    st.header("💸 Simulasi Jumlah Pinjaman")
    total_kebutuhan = st.slider("Total Kebutuhan Dana", 10_000_000, 5_000_000_000, 500_000_000, step=10_000_000)
    down_payment_pct = st.slider("Uang Muka (DP) (%)", 0, 100, 10)
    down_payment = total_kebutuhan * down_payment_pct / 100
    jumlah_pinjaman = total_kebutuhan - down_payment

    durasi_tahun = st.slider("Durasi Pinjaman (Tahun)", 1, 30, 5)
    bunga_efektif = st.number_input("Suku Bunga (eff. p.a.)", 0.0, 50.0, 10.0)

    r = bunga_efektif / 100 / 12
    n = durasi_tahun * 12
    angsuran_per_bulan = (r * jumlah_pinjaman) / (1 - (1 + r)**-n) if r > 0 else jumlah_pinjaman / n

    st.markdown("---")
    st.subheader("📊 Hasil Simulasi")
    st.write(f"**Total Pinjaman:** Rp {int(jumlah_pinjaman):,}")
    st.write(f"**Angsuran / Bulan:** Rp {int(angsuran_per_bulan):,}")

elif choice == "📋 Formulir Prediksi":
    st.header("📋 Form Pengajuan & Prediksi")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage','Other'])
            education = st.selectbox("Pendidikan Terakhir", list(predictor.education_map.keys()))
            previous_default = st.selectbox("Pernah Gagal Bayar?", list(predictor.default_map.keys()))
            credit_score = st.slider("Skor Kredit (300 - 850)", 300, 850, 650)

        with col2:
            income = st.slider("Pendapatan Tahunan ($)", 0, 500000, 100000, step=100)
            emp_exp = st.number_input("Lama Bekerja (Tahun)", 0)
            age = st.number_input("Umur", 18, 100, 35)
            cred_hist_len = st.number_input("Lama Riwayat Kredit (Tahun)", 0)
            loan_amount = st.slider("Jumlah Pinjaman ($)", 0, 100000, 10000, step=100)
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", 0.0, 100.0)
            loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_dict = {
            'person_income': income,
            'person_education': predictor.education_map[education],
            'loan_amnt': loan_amount,
            'loan_int_rate': loan_int_rate,
            'previous_loan_defaults_on_file': predictor.default_map[previous_default],
            'person_age': age,
            'person_emp_exp': emp_exp,
            'loan_percent_income': loan_amount / income if income > 0 else 0,
            'cb_person_cred_hist_length': cred_hist_len,
            'credit_score': credit_score,
            'person_gender_Male': 1 if gender == 'Male' else 0,
            'person_home_ownership_Own': 1 if home_ownership == 'Own' else 0,
            'person_home_ownership_Rent': 1 if home_ownership == 'Rent' else 0,
            'person_home_ownership_Other': 1 if home_ownership == 'Other' else 0,
            'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
            'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
            'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
            'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
            'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
            'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == 'DEBTCONSOLIDATION' else 0
        }

        prediction, probability = predictor.predict(input_dict)

        if prediction == 1:
            st.success("✅ Pinjaman kamu kemungkinan **DISETUJUI**! Selamat! 🎉")
        else:
            st.error("❌ Pinjaman kamu kemungkinan **DITOLAK**. Coba cek kembali detail pengajuanmu.")
