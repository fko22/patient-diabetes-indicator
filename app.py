import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model and scaler
model = joblib.load('models/random_forest_diabetes_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Feature names and descriptions
feature_info = {
    'HighBP': ("History of high blood pressure", "No"),
    'HighChol': ("History of high cholesterol", "No"),
    'CholCheck': ("Cholesterol check within last 5 years", "No"),
    'BMI': ("Body Mass Index(kg/m2)", 28),
    'Smoker': ("History of smoking", "No"),
    'Stroke': ("History of stroke", "No"),
    'HeartDiseaseorAttack': ("History of heart disease or heart attack", "No"),
    'PhysActivity': ("Physical activity in the last 30 days", "Yes"),
    'Fruits': ("Consumed fruits 1+ times per day", "Yes"),
    'Veggies': ("Consumed vegetables 1+ times per day", "Yes"),
    'HvyAlcoholConsump': ("Heavy alcohol consumption", "No"),
    'AnyHealthcare': ("Access to healthcare coverage", "Yes"),
    'NoDocbcCost': ("Couldn't see a doctor due to cost", "No"),
    'GenHlth': ("General health rating (1-5, where 1 = Excellent and 5 = Poor)", 3),
    'MentHlth': ("Number of mentally unhealthy days in the past 30 days", 5),
    'PhysHlth': ("Number of physically unhealthy days in the past 30 days", 5),
    'DiffWalk': ("Difficulty walking", "No"),
    'Sex': ("Gender (0 = Female, 1 = Male)", 1),
    'Age': ("Age group (1 = 18-24, ..., 13 = 80+)", 8),
    'Education': ("Level of education (1 = No education, ..., 6 = College graduate)", 4),
    'Income': ("Income level (1 = < $10,000, ..., 8 = $75,000+)", 6)
}

# Streamlit UI
st.title("Diabetes Prediction App with Probability")
st.write("Enter the following health indicator values to predict diabetes presence:")

# User input for each feature with Yes/No options or numeric input
user_input = []
for feature, (description, example_value) in feature_info.items():
    if example_value in ["Yes", "No"]:
        value = st.selectbox(f"{description} (Yes/No):", options=["No", "Yes"])
        user_input.append(1 if value == "Yes" else 0)
    else:
        value = st.number_input(f"{description}:", min_value=0, max_value=100, value=example_value)
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
