import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title("Employee Income Prediction")

# Collect user inputs
age = st.slider("Age", min_value=17, max_value=90, value=30)
hours_per_week = st.slider("Working Hours Per Week", min_value=1, max_value=100, value=40)

gender = st.selectbox("Gender", label_encoders['gender'].classes_)
native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)
occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
education = st.selectbox("Education", label_encoders['education'].classes_)

# Encode inputs
input_data = {
    'age': age,
    'hours-per-week': hours_per_week,
    'gender': label_encoders['gender'].transform([gender])[0],
    'native-country': label_encoders['native-country'].transform([native_country])[0],
    'occupation': label_encoders['occupation'].transform([occupation])[0],
    'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
    'workclass': label_encoders['workclass'].transform([workclass])[0],
    'education': label_encoders['education'].transform([education])[0],
}

features = np.array([[input_data[col] for col in [
    'age', 'gender', 'native-country', 'occupation',
    'marital-status', 'workclass', 'education', 'hours-per-week'
]]])

# Predict
if st.button("Predict Income"):
    prediction = model.predict(features)[0]
    income_label = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"Predicted Income Group: {income_label}")
