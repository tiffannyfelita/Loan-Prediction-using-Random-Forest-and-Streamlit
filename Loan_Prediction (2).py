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

# Setup
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction App")

# Load Data
data = pd.read_csv("Dataset_A_loan.csv")
st.subheader("Raw Dataset")
st.dataframe(data.head())

# Data Info
with st.expander("Data Summary"):
    st.write("Shape of dataset:", data.shape)
    st.write("Columns:", list(data.columns))
    st.write("Null values:", data.isnull().sum())

# Visualizations
st.subheader("Missing Data Visualization")
fig1 = plt.figure(figsize=(10,4))
mso.matrix(data, fontsize=12)
st.pyplot(fig1)

st.subheader("Correlation Heatmap")
fig2 = plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig2)

# Preprocessing
st.subheader("Preprocessing")
data.dropna(inplace=True)
x = data.drop('loan_default', axis=1)
y = data['loan_default']

scaler = MinMaxScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Balancing with SMOTE
smote = SMOTE()
x_bal, y_bal = smote.fit_resample(x_scaled, y)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

st.subheader("Model Evaluation")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

# SHAP Explanation (optional, simplified for demo)
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
st.sidebar.write("Loan Default" if pred[0] == 1 else "No Default")
