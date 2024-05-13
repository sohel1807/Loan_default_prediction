import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.title("Loan Default Prediction")
sc=pickle.load(open("scaler.pkl","rb"))
model=pickle.load(open("xg.pkl","rb"))
grade_mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}
home_ownership_mapping = {
    'MORTGAGE': 0,
    'OWN': 1,
    'RENT': 2
}
purpose_mapping = {
    'car': 0,
    'credit_card': 1,
    'debt_consolidation': 2,
    'home_improvement': 3,
    'house': 4,
    'major_purchase': 5,
    'medical': 6,
    'moving': 7,
    'other': 8,
    'small_business': 9,
    'vacation': 10,
    'wedding': 11
}
term_mapping = {
    ' 36 months': 0,
    ' 60 months': 1
}
col1,col2,col3 =st.columns(3)
with col1:
    grade=st.selectbox("Select Grade(A(very good)->G(very bad)):",list(grade_mapping.keys()))
with col2:
    annual_inc=st.number_input("Enter annual income")
with col3:
    emp_length_num=st.selectbox("Select employee length:",np.arange(1,12))
col4,col5,col6 =st.columns(3)
with col4:
    home_ownership=st.selectbox("Select home ownership:",list(home_ownership_mapping.keys()))
with col5:
    dti=st.number_input("Enter dti ratio")
with col6:
    purpose=st.selectbox("Select purpose",list(purpose_mapping.keys()))
col7,col8,col9,col10= st.columns(4)
with col7:
    term=st.selectbox("Select Term:",list(term_mapping.keys()))
with col8:
    last_delinq_none=st.selectbox("select last delinq",sorted([0,1]))
with col9:
    revol_util=st.number_input("Enter revol util")
col11,col12=st.columns(2)
with col11:
    total_rec_late_fee=st.number_input("rec late fee")
with col12:
    od_ratio=st.number_input("od_ratio")


# Encode categorical variables
grade_encoded = grade_mapping[grade]
home_ownership_encoded = home_ownership_mapping[home_ownership]
purpose_encoded = purpose_mapping[purpose]
term_encoded = term_mapping[term]

# Create a DataFrame with the input features
input_features = pd.DataFrame({
    'grade': [grade_encoded],
    'annual_inc': [annual_inc],
    'emp_length_num': [emp_length_num],
    'home_ownership': [home_ownership_encoded],
    'dti': [dti],
    'purpose': [purpose_encoded],
    'term': [term_encoded],
    'last_delinq_none': [last_delinq_none],
    'revol_util': [revol_util],
    'total_rec_late_fee': [total_rec_late_fee],
    'od_ratio': [od_ratio]
})

# Define the feature columns to scale
feature_cols = ['annual_inc', 'emp_length_num', 'total_rec_late_fee', 'dti', 'revol_util', 'od_ratio']

# Extract the subset of input features
input_features_subset = input_features[feature_cols]

# Scale numerical features
input_features_scaled_subset = sc.transform(input_features_subset)

# Replace the scaled features in the original DataFrame
input_features_scaled = input_features.copy()
input_features_scaled[feature_cols] = input_features_scaled_subset

# Make predictions using the model
prediction = model.predict(input_features_scaled)
if st.button("Loan Default Prediction"):
    # Display prediction
    if prediction[0] == 1:
        st.error("Prediction: Bad Loan")
    else:
        st.success("Prediction: Good Loan")

