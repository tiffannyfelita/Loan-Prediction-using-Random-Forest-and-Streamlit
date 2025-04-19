import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import shap
import pickle

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
    best_model = pickle.load(f)

# Load Data
data = pd.read_csv("Dataset_A_loan.csv")
st.subheader("Raw Dataset")
st.dataframe(data.head())

# Data Info
with st.expander("Data Summary"):
    st.write("Shape of dataset:", data.shape)
    st.write("Columns:", list(data.columns))
    st.write("Null values:", data.isnull().sum())

# Preprocessing
st.subheader("Preprocessing")
data.dropna(inplace=True)
data.replace(education_map, inplace=True)
x = data.drop('loan_default', axis=1)
y = data['loan_default']

scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Balancing with SMOTE
smote = SMOTE()
x_bal, y_bal = smote.fit_resample(x_scaled, y)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size=0.2, random_state=42)

# Model Training Option
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost (Pre-trained)"])
if model_choice == "Random Forest":
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
else:
    model = best_model

# Prediction and Evaluation
if model_choice == "Random Forest":
    y_pred = model.predict(x_test)

    st.subheader("Model Evaluation")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Feature Importance")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    fig3, ax = plt.subplots()
    shap.summary_plot(shap_values, x_test, show=False)
    st.pyplot(fig3)

# Prediction Form
st.sidebar.header("Make a Prediction")
user_input = {}
for col in x.columns:
    val = st.sidebar.slider(col, float(data[col].min()), float(data[col].max()))
    user_input[col] = val

user_df = pd.DataFrame([user_input])
user_df_scaled = scaler.transform(user_df)
pred = model.predict(user_df_scaled)

st.sidebar.subheader("Prediction Result")
st.sidebar.write(default_map_inv[pred[0]])
