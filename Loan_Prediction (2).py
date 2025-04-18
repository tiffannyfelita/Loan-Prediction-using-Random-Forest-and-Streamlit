import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# --- Load Model & Mapping ---
def load_model_and_maps():
    with open('best_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('education_map.pkl', 'rb') as f:
        education_map = pickle.load(f)
    with open('default_map.pkl', 'rb') as f:
        default_map = pickle.load(f)
    return model, education_map, default_map

model, education_map, default_map = load_model_and_maps()

# --- Custom UI Style ---
st.set_page_config(page_title="Prediksi Pinjaman", page_icon="💰", layout="centered")
st.markdown("""
    <style>
        html, body {
            background-color: #f7f9fb;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .predict-button {
            background-color: #3498db;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            border: none;
            transition: 0.3s;
        }
        .predict-button:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# --- Judul UI ---
st.markdown("<div class='main-title'>💳 Prediksi Kelolosan Pinjaman</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Masukkan informasi untuk melihat apakah pinjaman akan disetujui</div>", unsafe_allow_html=True)

# --- Form Input ---
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Usia", 18, 80, 25)
        emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 50, 5)
        cred_hist_len = st.slider("Lama Riwayat Kredit (tahun)", 0, 30, 5)
        credit_score = st.slider("Skor Kredit", 300, 850, 600)
        education = st.selectbox("Pendidikan Terakhir", list(education_map.keys()))
        previous_default = st.selectbox("Pernah Gagal Bayar?", list(default_map.keys()))

    with col2:
        income = st.number_input("Pendapatan Tahunan ($)", min_value=1, value=50000, step=1000)
        loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1, value=10000, step=500)
        loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
        gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
        home_ownership = st.selectbox("Status Tempat Tinggal", ['Own', 'Rent', 'Mortgage', 'Other'])
        loan_intent = st.selectbox("Tujuan Pinjaman", ['VENTURE', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

# --- Prediksi ---
if submitted:
    loan_percent_income = loan_amount / income

    input_dict = {
        "person_age": age,
        "person_education": education_map[education],
        "person_income": income,
        "person_emp_exp": emp_exp,
        "loan_amnt": loan_amount,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cred_hist_len,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": default_map[previous_default],
        "person_gender_male": 1 if gender == "Male" else 0,
        "person_home_ownership_OWN": 1 if home_ownership == "Own" else 0,
        "person_home_ownership_RENT": 1 if home_ownership == "Rent" else 0,
        "person_home_ownership_OTHER": 1 if home_ownership == "Other" else 0,
        "loan_intent_EDUCATION": 1 if loan_intent == "EDUCATION" else 0,
        "loan_intent_HOMEIMPROVEMENT": 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        "loan_intent_MEDICAL": 1 if loan_intent == "MEDICAL" else 0,
        "loan_intent_PERSONAL": 1 if loan_intent == "PERSONAL" else 0,
        "loan_intent_VENTURE": 1 if loan_intent == "VENTURE" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    expected_features = model.get_booster().feature_names
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.success("✅ Pinjaman kamu kemungkinan **DISETUJUI**! 🎉")
    else:
        st.error("❌ Pinjaman kamu kemungkinan **DITOLAK**. Coba cek ulang datamu.")

    # --- SHAP Explanation ---
    st.markdown("### 🔍 Penjelasan Model (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.markdown("Fitur mana yang paling berpengaruh terhadap keputusan model?")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=input_df.iloc[0]), max_display=12, show=False)
    st.pyplot(fig)
