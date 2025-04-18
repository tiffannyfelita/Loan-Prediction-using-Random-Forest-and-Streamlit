import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# --- Load model dan mapping ---
def load_model_and_maps():
    with open('best_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('education_map.pkl', 'rb') as f:
        education_map = pickle.load(f)
    with open('default_map.pkl', 'rb') as f:
        default_map = pickle.load(f)
    return model, education_map, default_map

model, education_map, default_map = load_model_and_maps()

# --- UI Styling ---
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown("""
    <style>
    .title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💳 Prediksi Kelolosan Pinjaman</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Isi informasi kamu di bawah ini untuk melihat apakah pinjaman akan disetujui</div>", unsafe_allow_html=True)

# --- Form Input ---
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Usia", 18, 80, 30)
        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        education = st.selectbox("Pendidikan Terakhir", list(education_map.keys()))
        income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=50000, step=1000)
        emp_exp = st.slider("Lama Bekerja (tahun)", 0, 40, 5)
        credit_score = st.slider("Skor Kredit (300–850)", 300, 850, 650)
        previous_default = st.selectbox("Pernah Gagal Bayar?", list(default_map.keys()))

    with col2:
        home_ownership = st.selectbox("Status Tempat Tinggal", ['Own', 'Rent', 'Mortgage', 'Other'])
        loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1000, value=10000, step=500)
        loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
        loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
        cred_hist_len = st.slider("Lama Riwayat Kredit (tahun)", 0, 30, 8)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- Prediksi & Output ---
if submitted:
    input_dict = {
        'person_age': age,
        'person_gender': gender,
        'person_education': education_map[education],
        'person_income': income,
        'person_emp_exp': emp_exp,
        'person_home_ownership': home_ownership,
        'loan_amnt': loan_amount,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_amount / income,
        'cb_person_cred_hist_length': cred_hist_len,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': default_map[previous_default],
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("✅ Pinjaman kamu kemungkinan **DISETUJUI**! 🎉")
    else:
        st.error("❌ Pinjaman kamu kemungkinan **DITOLAK**.")

    # --- SHAP Explanation ---
    st.markdown("### 🔍 Penjelasan Prediksi (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0]
        ),
        show=False
    )
    st.pyplot(fig)
