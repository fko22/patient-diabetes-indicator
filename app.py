import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model and scaler
model = joblib.load('models/random_forest_diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature names and descriptions
feature_info = {
    'HighBP': ("History of high blood pressure (0 = No, 1 = Yes)", 1),
    'HighChol': ("History of high cholesterol (0 = No, 1 = Yes)", 1),
    'CholCheck': ("Cholesterol check within last 5 years (0 = No, 1 = Yes)", 1),
    'BMI': ("Body Mass Index", 28),
    'Smoker': ("History of smoking (0 = No, 1 = Yes)", 0),
    'Stroke': ("History of stroke (0 = No, 1 = Yes)", 0),
    'HeartDiseaseorAttack': ("History of heart disease or heart attack (0 = No, 1 = Yes)", 0),
    'PhysActivity': ("Physical activity in the last 30 days (0 = No, 1 = Yes)", 1),
    'Fruits': ("Consumed fruits 1+ times per day (0 = No, 1 = Yes)", 1),
    'Veggies': ("Consumed vegetables 1+ times per day (0 = No, 1 = Yes)", 1),
    'HvyAlcoholConsump': ("Heavy alcohol consumption (0 = No, 1 = Yes)", 0),
    'AnyHealthcare': ("Access to healthcare coverage (0 = No, 1 = Yes)", 1),
    'NoDocbcCost': ("Couldn't see a doctor due to cost (0 = No, 1 = Yes)", 0),
    'GenHlth': ("General health rating (1-5, where 1 = Excellent and 5 = Poor)", 3),
    'MentHlth': ("Number of mentally unhealthy days in the past 30 days", 5),
    'PhysHlth': ("Number of physically unhealthy days in the past 30 days", 5),
    'DiffWalk': ("Difficulty walking (0 = No, 1 = Yes)", 0),
    'Sex': ("Gender (0 = Female, 1 = Male)", 1),
    'Age': ("Age group (1 = 18-24, ..., 13 = 80+)", 8),
    'Education': ("Level of education (1 = No education, ..., 6 = College graduate)", 4),
    'Income': ("Income level (1 = < $10,000, ..., 8 = $75,000+)", 6)
}

# Streamlit UI
st.title("Diabetes Prediction App with Probability")
st.write("Enter the following health indicator values to predict diabetes presence:")

# User input for each feature with prefilled values
user_input = []
for feature, (description, example_value) in feature_info.items():
    value = st.number_input(f"{feature} ({description}):", min_value=0, max_value=100, value=example_value)
    user_input.append(value)

# Convert user input to a numpy array and reshape to 2D
user_input_array = np.array([user_input]).reshape(1, -1)

# Scale the input features using the loaded scaler
scaled_input = scaler.transform(user_input_array)

# Predict button
if st.button("Predict"):
    # Get the prediction probabilities
    probabilities = model.predict_proba(scaled_input)[0]
    diabetes_prob = probabilities[1]  # Probability of class 1 (Diabetes Present)
    no_diabetes_prob = probabilities[0]  # Probability of class 0 (No Diabetes)

    # Display the prediction result with probabilities
    if diabetes_prob > no_diabetes_prob:
        st.success(f"The model predicts: **Diabetes Present** with {diabetes_prob*100:.2f}% probability")
    else:
        st.success(f"The model predicts: **No Diabetes** with {no_diabetes_prob*100:.2f}% probability")
