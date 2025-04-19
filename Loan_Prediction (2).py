import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the model and encoding mappings
@st.cache_resource
def load_model():
    with open("best_xgb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mappings():
    with open("education_map.pkl", "rb") as f:
        education_map = pickle.load(f)
    with open("default_map.pkl", "rb") as f:
        default_map = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:  # Add this line
        feature_names = pickle.load(f)
    return education_map, default_map, feature_names

model = load_model()
education_map, default_map, feature_names = load_mappings()

# Create the Streamlit app
st.title("Loan Approval Prediction System")

st.write("""
This application predicts the likelihood of a loan being approved based on applicant information.
Please fill in the details below and click 'Predict' to see the result.
""")

# Input fields (same as before)
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Gender", options=['Female', 'Male'])  # Fixed typo in variable name
    person_education = st.selectbox("Education Level", options=list(education_map.keys()))
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=5)
    
with col2:
    person_home_ownership = st.selectbox("Home Ownership", options=['Rent', 'Own', 'Mortgage', 'Other'])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=20000)
    loan_intent = st.selectbox("Loan Purpose", options=['Personal', 'Education', 'Medical', 'Venture', 'Home', 'Debt'])
    loan_int_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    previous_loan_defaults_on_file = st.selectbox("Previous Defaults", options=list(default_map.keys()))

# Calculate loan percent income
loan_percent_income = (loan_amnt / person_income) * 100 if person_income > 0 else 0
st.write(f"Loan Percentage of Income: {loan_percent_income:.2f}%")

# Prediction button
if st.button("Predict Loan Approval"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_home_ownership': [person_home_ownership],
        'loan_intent': [loan_intent],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })
    
    # Apply encoding from pickle files
    input_data['person_education'] = input_data['person_education'].map(education_map)
    input_data['previous_loan_defaults_on_file'] = input_data['previous_loan_defaults_on_file'].map(default_map)
    
    # One-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data, columns=['person_gender', 'person_home_ownership', 'loan_intent'], drop_first=True)
    
    # Create a DataFrame with all expected columns initialized to 0
    final_input = pd.DataFrame(columns=feature_names)
    final_input.loc[0] = 0  # Initialize all values to 0
    
    # Fill in the values we have
    for col in input_data.columns:
        if col in final_input.columns:
            final_input[col] = input_data[col]
    
    # Ensure correct data types
    final_input = final_input.astype(float)
    
    # Make prediction
    try:
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)
        
        # Display results
        if prediction[0] == 1:
            st.error("Loan Application: **Denied**")
            st.write(f"Probability of denial: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success("Loan Application: **Approved**")
            st.write(f"Probability of approval: {prediction_proba[0][0]*100:.2f}%")
        
        # Show feature importance if available
        try:
            st.subheader("Key Factors Influencing Decision")
            feature_importance = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(feature_importance.set_index('Feature'))
        except AttributeError:
            pass
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check that all input values are valid and try again.")

# Add some explanations
st.markdown("""
### About the Model
- This prediction is based on an XGBoost machine learning model trained on historical loan data.
- The model considers various factors including credit history, income, loan amount, and more.
- Predictions are estimates only and should be used as one of several decision-making tools.
""")
