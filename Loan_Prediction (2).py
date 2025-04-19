import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Setup
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction App")

# Load mappings and pre-trained model
with open("education_map.pkl", "rb") as f:
    education_map = pickle.load(f)

with open("default_map.pkl", "rb") as f:
    default_map = pickle.load(f)
    default_map_inv = {v: k for k, v in default_map.items()}

with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Data
data = pd.read_csv("Dataset_A_loan.csv")

# Preprocessing
data.dropna(inplace=True)
data.replace(education_map, inplace=True)

# Determine target column safely
target_col = [col for col in data.columns if 'loan' in col.lower() and 'default' in col.lower()]
if not target_col:
    st.error("Kolom target 'loan_default' tidak ditemukan.")
    st.stop()
else:
    target_col = target_col[0]

x = data.drop(target_col, axis=1)
x = pd.get_dummies(x)
scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# User Input (Main Area)
st.subheader("Input Features")
user_input = {}
for col in x.columns:
    val = st.slider(col, float(x[col].min()), float(x[col].max()))
    user_input[col] = val

user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)
pred = model.predict(user_df_scaled)

# Prediction Result
st.subheader("Prediction Result")
st.write(f"**Prediction:** {default_map_inv[pred[0]]}")
