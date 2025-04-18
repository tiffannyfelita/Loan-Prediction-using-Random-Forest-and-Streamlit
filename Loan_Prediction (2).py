import streamlit as st
import pandas as pd
import pickle

# === Multi-page Layout ===
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")

# === Sidebar Navigation ===
menu = ["Prediksi", "Informasi Fitur"]
choice = st.sidebar.selectbox("Navigasi", menu)

# === Load Model & Mapping ===
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("education_map.pkl", "rb") as f:
    education_map = pickle.load(f)

with open("default_map.pkl", "rb") as f:
    default_map = pickle.load(f)

if choice == "Prediksi":
    st.title("💳 Prediksi Kelolosan Pinjaman")
    st.markdown("Masukkan informasi lengkap untuk memprediksi apakah pinjaman akan disetujui.")

    with st.form("loan_form"):
        st.slider("Umur Pemohon", 18, 100, 30, key="person_age")
        st.slider("Lama Bekerja (tahun)", 0, 40, 5, key="person_emp_exp")
        st.slider("Panjang Riwayat Kredit (tahun)", 0, 30, 3, key="cb_person_cred_hist_length")
        st.slider("Skor Kredit (300-850)", 300, 850, 600, key="credit_score")

        st.selectbox("Jenis Kelamin", ['Male', 'Female'], key="gender")
        st.selectbox("Status Tempat Tinggal", ['Rent', 'Own', 'Mortgage'], key="home")
        st.selectbox("Pendidikan Terakhir", list(education_map.keys()), key="education")
        st.selectbox("Pernah Gagal Bayar?", list(default_map.keys()), key="default")

        st.number_input("Pendapatan Tahunan ($)", min_value=0, step=1000, key="income")
        st.number_input("Jumlah Pinjaman ($)", min_value=0, step=1000, key="loan_amount")
        st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, step=0.1, key="loan_int_rate")

        st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], key="loan_intent")

        submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
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
            'person_gender_Male': 1 if st.session_state.gender == 'Male' else 0,
            'person_home_ownership_Own': 1 if st.session_state.home == 'Own' else 0,
            'person_home_ownership_Rent': 1 if st.session_state.home == 'Rent' else 0,
            'loan_intent_EDUCATION': 1 if st.session_state.loan_intent == 'EDUCATION' else 0,
            'loan_intent_HOMEIMPROVEMENT': 1 if st.session_state.loan_intent == 'HOMEIMPROVEMENT' else 0,
            'loan_intent_MEDICAL': 1 if st.session_state.loan_intent == 'MEDICAL' else 0,
            'loan_intent_PERSONAL': 1 if st.session_state.loan_intent == 'PERSONAL' else 0,
            'loan_intent_VENTURE': 1 if st.session_state.loan_intent == 'VENTURE' else 0,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=model.get_booster().feature_names, fill_value=0)
        prediction = model.predict(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.success("✅ Pinjaman kamu kemungkinan DISETUJUI! 🎉")
        else:
            st.error("❌ Pinjaman kamu kemungkinan DITOLAK. Coba cek kembali detail pengajuanmu.")

elif choice == "Informasi Fitur":
    st.title("📘 Informasi Fitur-Fitur Prediksi Pinjaman")
    st.markdown("Gunakan tabel di bawah ini untuk memahami arti dari masing-masing fitur dalam sistem prediksi pinjaman.")

    feature_data = [
        ("person_age", "Usia Pemohon", "Umur dari pemohon pinjaman dalam satuan tahun."),
        ("person_gender", "Jenis Kelamin", "Gender pemohon: Male atau Female."),
        ("person_education", "Pendidikan Terakhir", "Tingkat pendidikan tertinggi yang dimiliki pemohon."),
        ("person_income", "Pendapatan Tahunan", "Jumlah pendapatan per tahun dalam dolar."),
        ("person_emp_exp", "Pengalaman Kerja", "Lama pengalaman kerja pemohon dalam tahun."),
        ("person_home_ownership", "Status Tempat Tinggal", "Jenis kepemilikan tempat tinggal: Own, Rent, Mortgage, atau Other."),
        ("loan_amnt", "Jumlah Pinjaman", "Total nilai pinjaman yang diminta pemohon."),
        ("loan_intent", "Tujuan Pinjaman", "Alasan atau tujuan dari pinjaman seperti EDUCATION, MEDICAL, dll."),
        ("loan_int_rate", "Bunga Pinjaman", "Tingkat suku bunga tahunan yang berlaku untuk pinjaman."),
        ("loan_percent_income", "Rasio Pinjaman terhadap Pendapatan", "Persentase pinjaman dibanding pendapatan tahunan: loan_amnt / income."),
        ("cb_person_cred_hist_length", "Lama Riwayat Kredit", "Panjang riwayat kredit pemohon dalam tahun."),
        ("credit_score", "Skor Kredit", "Nilai numerik (300–850) yang menunjukkan kelayakan kredit."),
        ("previous_loan_defaults_on_file", "Pernah Gagal Bayar", "Apakah pemohon pernah memiliki tunggakan pinjaman sebelumnya."),
        ("loan_status", "Status Pinjaman (Target)", "Apakah pinjaman disetujui (1) atau ditolak (0).")
    ]

    st.markdown("### 🧾 Tabel Penjelasan Fitur")
    st.table(pd.DataFrame(feature_data, columns=["Kode Fitur", "Nama Fitur", "Deskripsi"]))

    with st.expander("❓ Apa itu Skor Kredit?"):
        st.markdown("""
        **Skor Kredit** adalah angka dari 300–850 yang menunjukkan tingkat kepercayaan lembaga keuangan terhadap peminjam:

        - Skor tinggi berarti riwayat keuangan baik dan kemungkinan disetujui lebih besar.
        - Skor rendah berarti risiko gagal bayar tinggi dan pinjaman cenderung ditolak.

        Biasanya dihitung dari:
        - Riwayat pembayaran
        - Lama kredit aktif
        - Jenis pinjaman
        - Rasio penggunaan kredit
        - Permohonan kredit baru
        """)
