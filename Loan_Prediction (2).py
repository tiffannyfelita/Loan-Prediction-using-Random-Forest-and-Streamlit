import streamlit as st
import pandas as pd
import pickle

class LoanPredictor:
    def __init__(self):
        self.load_model()
        self.load_mappings()

    def load_model(self):
        with open('best_xgb_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def load_mappings(self):
        with open('education_map.pkl', 'rb') as f:
            self.education_map = pickle.load(f)
        with open('default_map.pkl', 'rb') as f:
            self.default_map = pickle.load(f)

    def predict(self, data):
        df = pd.DataFrame([data])
        expected_features = self.model.get_booster().feature_names
        df = df.reindex(columns=expected_features, fill_value=0)
        result = self.model.predict(df)[0]
        return result

    def show_prediction_form(self):
        st.subheader("📋 Formulir Pengajuan Pinjaman")

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'])
            education = st.selectbox("Pendidikan Terakhir", list(self.education_map.keys()))
            previous_default = st.selectbox("Pernah Gagal Bayar?", ['No', 'Yes'])

        with col2:
            income = st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000)
            loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000)
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1)
            loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

        if st.button("🔍 Prediksi Sekarang"):
            input_data = {
                'person_income': income,
                'person_education': self.education_map[education],
                'loan_amnt': loan_amount,
                'loan_int_rate': loan_int_rate,
                'previous_loan_defaults_on_file': self.default_map[previous_default],

                'person_gender_Male': 1 if gender == 'Male' else 0,
                'person_home_ownership_Own': 1 if home_ownership == 'Own' else 0,
                'person_home_ownership_Rent': 1 if home_ownership == 'Rent' else 0,
                'person_home_ownership_Mortgage': 1 if home_ownership == 'Mortgage' else 0,

                'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
                'loan_intent_HOMEIMPROVEMENT': 1 if loan_intent == 'HOMEIMPROVEMENT' else 0,
                'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
                'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0,
                'loan_intent_VENTURE': 1 if loan_intent == 'VENTURE' else 0,
                'loan_intent_DEBTCONSOLIDATION': 1 if loan_intent == 'DEBTCONSOLIDATION' else 0,
            }

            prediction = self.predict(input_data)
            st.markdown("---")
            if prediction == 1:
                st.success("✅ Pinjaman kamu kemungkinan **DISETUJUI**! Selamat! 🎉")
            else:
                st.error("❌ Pinjaman kamu kemungkinan **DITOLAK**. Coba cek kembali detail pengajuanmu.")

# --- Main Streamlit App ---
def main():
    st.title("💰 Aplikasi Prediksi Pinjaman")
    menu = ["🏠 Beranda", "🔮 Prediksi"]
    choice = st.sidebar.selectbox("Navigasi", menu)

    app = LoanPredictor()

    if choice == "🏠 Beranda":
        st.subheader("Selamat datang di aplikasi prediksi kelolosan pinjaman!")
        st.write("Gunakan menu di samping untuk memulai.")
    elif choice == "🔮 Prediksi":
        app.show_prediction_form()

if __name__ == '__main__':
    main()
