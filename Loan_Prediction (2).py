import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class LoanPredictor:
    def __init__(self):
        self.model = None
        self.education_map = {}
        self.default_map = {}
        self.load_all()

    def load_all(self):
        st.write("🚀 Memuat model dan mapping...")

        try:
            with open('best_xgb_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
                st.success("✅ Model berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ Gagal load model: {type(e).__name__} - {e}")

        try:
            with open('education_map.pkl', 'rb') as f:
                self.education_map = pickle.load(f)
                st.success("✅ Mapping pendidikan berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ Gagal load education_map: {type(e).__name__} - {e}")

        try:
            with open('default_map.pkl', 'rb') as f:
                self.default_map = pickle.load(f)
                st.success("✅ Mapping gagal bayar berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ Gagal load default_map: {type(e).__name__} - {e}")

    def predict(self, data):
        if self.model is None:
            st.error("Model belum dimuat. Tidak bisa melakukan prediksi.")
            return None, None

        df = pd.DataFrame([data])
        expected_features = self.model.get_booster().feature_names
        st.write("📋 Fitur yang diharapkan model:", expected_features)
        df = df.reindex(columns=expected_features, fill_value=0)
        result = self.model.predict(df)[0]
        return result, df

    def show_prediction_form(self):
        st.subheader("📋 Formulir Pengajuan Pinjaman")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
            home_ownership = st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'])

            if self.education_map:
                education = st.selectbox("Pendidikan Terakhir", list(self.education_map.keys()))
            else:
                education = st.text_input("Pendidikan Terakhir (teks manual)")
            
            if self.default_map:
                previous_default = st.selectbox("Pernah Gagal Bayar?", list(self.default_map.keys()))
            else:
                previous_default = st.text_input("Gagal Bayar? (Yes/No)")

        with col2:
            income = st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000)
            loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000)
            loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1)
            loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

        if st.button("🔍 Prediksi Sekarang"):
            try:
                input_data = {
                    'person_income': income,
                    'person_education': self.education_map.get(education, 0),
                    'loan_amnt': loan_amount,
                    'loan_int_rate': loan_int_rate,
                    'previous_loan_defaults_on_file': self.default_map.get(previous_default, 0),
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

                st.write("📦 Data untuk prediksi:")
                st.json(input_data)

                prediction, df = self.predict(input_data)
                if prediction is not None:
                    st.markdown("---")
                    if prediction == 1:
                        st.success("✅ Pinjaman kemungkinan **DISETUJUI**!")
                    else:
                        st.error("❌ Pinjaman kemungkinan **DITOLAK**.")
            except Exception as e:
                st.error(f"⚠️ Terjadi error saat prediksi: {type(e).__name__} - {e}")

    def show_visualization(self):
        st.subheader("📊 Visualisasi Data Pinjaman")
        try:
            df = pd.read_csv("data_pinjaman.csv")  # Ganti sesuai nama dataset kamu
            st.success("✅ Dataset berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ Gagal memuat dataset: {type(e).__name__} - {e}")
            return

        option = st.selectbox("Pilih kolom untuk distribusi:", ['person_income', 'loan_amnt', 'loan_int_rate'])

        fig, ax = plt.subplots()
        sns.histplot(df[option], bins=30, kde=True, color="skyblue", ax=ax)
        ax.set_title(f"Distribusi {option}")
        ax.set_xlabel(option)
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

        st.markdown("#### 🔎 Distribusi Tujuan Pinjaman")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="loan_intent", data=df, order=df["loan_intent"].value_counts().index, palette="viridis", ax=ax2)
        ax2.set_title("Distribusi Tujuan Pinjaman")
        ax2.set_ylabel("Jumlah")
        ax2.set_xlabel("Loan Intent")
        plt.xticks(rotation=30)
        st.pyplot(fig2)

# --- Main Streamlit App ---
def main():
    st.title("🛠 Debug Mode - Prediksi Pinjaman")
    menu = ["🏠 Home", "📋 Form Prediksi", "📊 Visualisasi Data"]
    choice = st.sidebar.selectbox("Navigasi", menu)

    app = LoanPredictor()

    if choice == "🏠 Home":
        st.markdown("Selamat datang di mode debug. Periksa log dan mapping file sebelum lanjut.")
    elif choice == "📋 Form Prediksi":
        app.show_prediction_form()
    elif choice == "📊 Visualisasi Data":
        app.show_visualization()

if __name__ == '__main__':
    main()
