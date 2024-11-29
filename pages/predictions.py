import pandas as pd
import shap
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from utils import explain_model, generate_recommendations, load_api_key
import sqlite3

conn = sqlite3.connect('user_predictions.db')
cursor = conn.cursor()

def save_prediction_to_db(user_id, user_input, prediction_result, diabetes_prob):
    try:
        today_date = date.today().strftime('%Y-%m-%d')  # Format as YYYY-MM-DD

        # Check if an entry exists for this user and date
        cursor.execute('''
        SELECT id FROM predictions 
        WHERE user_id = ? AND date = ?
        ''', (user_id, today_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # Update the existing entry
            cursor.execute('''
            UPDATE predictions
            SET 
                HighBP = ?, HighChol = ?, CholCheck = ?, BMI = ?, Smoker = ?, Stroke = ?,
                HeartDiseaseorAttack = ?, PhysActivity = ?, Fruits = ?, Veggies = ?,
                HvyAlcoholConsump = ?, AnyHealthcare = ?, NoDocbcCost = ?, GenHlth = ?, 
                MentHlth = ?, PhysHlth = ?, DiffWalk = ?, Sex = ?, Age = ?, 
                Education = ?, Income = ?, Prediction = ?, Probability = ?
            WHERE id = ?
            ''', (
                user_input['HighBP'], user_input['HighChol'], user_input['CholCheck'], user_input['BMI'],
                user_input['Smoker'], user_input['Stroke'], user_input['HeartDiseaseorAttack'],
                user_input['PhysActivity'], user_input['Fruits'], user_input['Veggies'],
                user_input['HvyAlcoholConsump'], user_input['AnyHealthcare'], user_input['NoDocbcCost'],
                user_input['GenHlth'], user_input['MentHlth'], user_input['PhysHlth'],
                user_input['DiffWalk'], user_input['Sex'], user_input['Age'],
                user_input['Education'], user_input['Income'], prediction_result, diabetes_prob,
                existing_entry[0]
            ))
        else:
            # Insert a new entry
            cursor.execute('''
            INSERT INTO predictions (
                user_id, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, 
                HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, 
                MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income, 
                Prediction, Probability, date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_input['HighBP'], user_input['HighChol'], user_input['CholCheck'], user_input['BMI'],
                user_input['Smoker'], user_input['Stroke'], user_input['HeartDiseaseorAttack'],
                user_input['PhysActivity'], user_input['Fruits'], user_input['Veggies'],
                user_input['HvyAlcoholConsump'], user_input['AnyHealthcare'], user_input['NoDocbcCost'],
                user_input['GenHlth'], user_input['MentHlth'], user_input['PhysHlth'],
                user_input['DiffWalk'], user_input['Sex'], user_input['Age'],
                user_input['Education'], user_input['Income'],
                prediction_result, diabetes_prob, today_date
            ))

        conn.commit()
        st.success("Prediction saved successfully!")
        conn.close() 
    except Exception as e:
        st.error(f"Error saving to database: {e}")

# Load the trained Random Forest model and scaler
model = joblib.load('./models/random_forest_diabetes_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Feature names and descriptions
feature_info = {
    'HighBP': ("History of high blood pressure", "No"),
    'HighChol': ("History of high cholesterol", "No"),
    'CholCheck': ("Cholesterol check within last 5 years", "No"),
    'BMI': ("Body Mass Index(kg/m2)", 28),
    'Smoker': ("Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]", "No"),
    'Stroke': ("(Ever told) you had a stroke", "No"),
    'HeartDiseaseorAttack': ("History of heart disease or heart attack", "No"),
    'PhysActivity': ("Physical activity in past 30 days - not including job", "Yes"),
    'Fruits': ("Consumed fruits 1+ times per day", "Yes"),
    'Veggies': ("Consumed vegetables 1+ times per day", "Yes"),
    'HvyAlcoholConsump': ("Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)", "No"),
    'AnyHealthcare': ("Access to healthcare coverage", "Yes"),
    'NoDocbcCost': ("Couldn't see a doctor due to cost", "No"),
    'GenHlth': ("How would rate your general health?", 3),
    'MentHlth': ("Number of mentally unhealthy days in the past 30 days", 0),
    'PhysHlth': ("Number of physically unhealthy days in the past 30 days", 0),
    'DiffWalk': ("Do you have serious difficulty walking or climbing stairs?", "No"),
    'Sex': ("Gender (0 = Female, 1 = Male)", 1),
    'Age': ("Age group (1 = 18-24, ..., 13 = 80+)", 8),
    'Education': ("Level of education (1 = No education, ..., 6 = College graduate)", 4),
    'Income': ("Income level (1 = < $10,000, ..., 8 = $75,000+)", 6)
}

# Streamlit UI
st.title("🌿 HealthTrack Diabetes 🌿")
st.warning("Predicting Diabetes Risk and Providing Lifestyle Recommendations")
# st.title("Diabetes Prediction and Lifestyle Recommendations")

# Demo Templates
demo_profiles = {
    "Sample Profile 1: Older Male with High BMI and Unhealthy Lifestyle": {
        'HighBP': 1,
        'HighChol': 1,
        'CholCheck': 0,
        'BMI': 35,
        'Smoker': 1,
        'Stroke': 1,
        'HeartDiseaseorAttack': 1,
        'PhysActivity': 0,
        'Fruits': 0,
        'Veggies': 0,
        'HvyAlcoholConsump': 1,
        'AnyHealthcare': 0,
        'NoDocbcCost': 1,
        'GenHlth': 5,
        'MentHlth': 20,
        'PhysHlth': 25,
        'DiffWalk': 1,
        'Sex': 1,  # Male
        'Age': 65,  # 80+
        'Education': 3,  # Some high school
        'Income': 2  # $10,000 - $15,000
    },
    "Sample Profile 2: Younger Male with High BMI and Unhealthy Lifestyle": {
        'HighBP': 0,
        'HighChol': 1,
        'CholCheck': 1,
        'BMI': 32,
        'Smoker': 1,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 0,
        'Fruits': 0,
        'Veggies': 0,
        'HvyAlcoholConsump': 1,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 4,
        'MentHlth': 15,
        'PhysHlth': 10,
        'DiffWalk': 0,
        'Sex': 1,  # Male
        'Age': 25,  # 25-29
        'Education': 4,  # High school graduate
        'Income': 5  # $25,000 - $35,000
    },
    "Sample Profile 3: Older Under Educated Female with Healthy Diet": {
        'HighBP': 1,
        'HighChol': 0,
        'CholCheck': 1,
        'BMI': 22,
        'Smoker': 0,
        'Stroke': 0,
        'HeartDiseaseorAttack': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'Veggies': 1,
        'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1,
        'NoDocbcCost': 0,
        'GenHlth': 2,
        'MentHlth': 2,
        'PhysHlth': 3,
        'DiffWalk': 0,
        'Sex': 0,  # Female
        'Age': 74,  # 70-74
        'Education': 1,  # No formal education
        'Income': 1  # < $10,000
    }
}

# Use a radio button for better visual clarity
selected_demo = st.radio(
    "Select a Demo Profile (optional) or enter custom values:",
    ["Custom"] + list(demo_profiles.keys())
)



# Initialize user_input as a dictionary to ensure correct size and feature mapping
if selected_demo != "Custom":
    user_input = demo_profiles[selected_demo]  # Load demo profile
else:
    st.error("Enter the following health indicator values to predict diabetes presence:")
    user_input = {feature: None for feature in feature_info.keys()}  # Empty values for manual entry

# Generate inputs based on demo or manual entry
for feature, (description, example_value) in feature_info.items():
    if feature == "BMI" and selected_demo == "Custom":
        weight = st.number_input("Weight (kg):", min_value=0.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (m):", min_value=0.0, max_value=3.0, value=1.75)
        bmi = weight / (height ** 2) if height > 0 else 0
        user_input[feature] = bmi
        st.write(f"Your calculated BMI: {bmi:.2f}")

    elif feature == "GenHlth":
        genhlth_options = {
            "Excellent": 1,
            "Very Good": 2,
            "Good": 3,
            "Fair": 4,
            "Poor": 5
        }
        genhlth_text = st.selectbox(
            "How would you rate your general health?",
            options=list(genhlth_options.keys()),
            index=demo_profiles[selected_demo].get(feature, 3) - 1 if selected_demo != "Custom" else 2
        )
        user_input[feature] = genhlth_options[genhlth_text]

    elif feature == "Sex":
        sex = st.selectbox(
            "Sex:", options=["Female", "Male"], index=demo_profiles[selected_demo].get(feature, 1) if selected_demo != "Custom" else 1
        )
        user_input[feature] = 1 if sex == "Male" else 0

    elif feature == "Age":
        age = st.number_input(
            "Age (years):",
            min_value=0, max_value=120,
            value=demo_profiles[selected_demo].get(feature, 25) if selected_demo != "Custom" else 25
        )
        user_input[feature] = (
            1 if age <= 24 else
            2 if age <= 29 else
            3 if age <= 34 else
            4 if age <= 39 else
            5 if age <= 44 else
            6 if age <= 49 else
            7 if age <= 54 else
            8 if age <= 59 else
            9 if age <= 64 else
            10 if age <= 69 else
            11 if age <= 74 else
            12 if age <= 79 else 13
        )

    elif feature == "Income":
        income = st.number_input(
            "Income (in $):",
            min_value=0, max_value=1000000,
            value=demo_profiles[selected_demo].get(feature, 50000) if selected_demo != "Custom" else 50000
        )
        user_input[feature] = (
            1 if income < 10000 else
            2 if income < 15000 else
            3 if income < 20000 else
            4 if income < 25000 else
            5 if income < 35000 else
            6 if income < 50000 else
            7 if income < 75000 else 8
        )

    elif feature == "Education":
        education_options = {
            "Never attended school or only kindergarten": 1,
            "Grades 1 through 8 (Elementary)": 2,
            "Grades 9 through 11 (Some high school)": 3,
            "Grade 12 or GED (High school graduate)": 4,
            "College 1 year to 3 years (Some college or technical school)": 5,
            "College 4 years or more (College graduate)": 6
        }
        education_text = st.selectbox(
            "Education Level:",
            options=list(education_options.keys()),
            index=demo_profiles[selected_demo].get(feature, 4) - 1 if selected_demo != "Custom" else 3
        )
        user_input[feature] = education_options[education_text]

    elif example_value in ["Yes", "No"]:
        value = st.selectbox(
            f"{description} (No/Yes):",
            options=["No", "Yes"],
            index=demo_profiles[selected_demo].get(feature, 0) if selected_demo != "Custom" else 0
        )
        user_input[feature] = 1 if value == "Yes" else 0

    else:
        value = st.number_input(
            f"{description}:",
            min_value=0, max_value=100,
            value=demo_profiles[selected_demo].get(feature, example_value) if selected_demo != "Custom" else example_value
        )
        user_input[feature] = value

# show input values as dataframe
# st.write(pd.DataFrame(user_input, index=[0]))

# Convert user input to a numpy array and reshape to 2D
user_input_array = np.array(list(user_input.values())).reshape(1, -1)

# Scale the input features using the loaded scaler
scaled_input = scaler.transform(user_input_array)


# Initialize session state for prediction_made
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False

# Predict button
if st.button("Predict"):
    # Get the prediction probabilities
    probabilities = model.predict_proba(scaled_input)[0]
    diabetes_prob = probabilities[1]  # Probability of class 1 (Diabetes Present)
    no_diabetes_prob = probabilities[0]  # Probability of class 0 (No Diabetes)

    # Determine the prediction result
    prediction_result = "Diabetes Present" if diabetes_prob > no_diabetes_prob else "No Diabetes Present"

    user_model_text = f"The model predicts: **{prediction_result}** with {max(diabetes_prob, no_diabetes_prob) * 100:.2f}% probability"
    save_prediction_to_db(st.session_state.user_id,user_input, prediction_result, diabetes_prob)

    # Display the prediction result with probabilities
    st.success(user_model_text)

    # Save prediction result in session state
    st.session_state.prediction_made = True
    st.session_state.user_model_text = user_model_text

    # Get SHAP values and feature contributions
    shap_values, feature_contributions = explain_model(
        feature_names=list(feature_info.keys()),
        X_sample=user_input_array,
        X_sample_scaled=scaled_input,
        rf_model=model
    )

    # Save SHAP values and contributions in session state
    st.session_state.shap_values = shap_values
    st.session_state.feature_contributions = feature_contributions

llm, api_key = load_api_key()

# Show additional options only if prediction has been made
if st.session_state.prediction_made and api_key is not None:
    # Generate LLM recommendations
    if st.button("Generate Personalized Recommendations"):
        st.write("### Personalized Lifestyle Recommendations:")
        recommendations = generate_recommendations(
            st.session_state.feature_contributions,
            feature_info,
            st.session_state.user_model_text,
            llm
        )

        # Display the "content" part of the response
        st.write(recommendations.content)

    with st.expander("View SHAP Values", expanded=False):
        # Visualize SHAP values using a bar plot
        st.write("### SHAP Feature Importance:")

        fig, ax = plt.subplots()
        shap.bar_plot(
            st.session_state.shap_values[0][:, 1],  # SHAP values for class 1 (diabetes) for the first sample
            feature_names=list(feature_info.keys()),
            show=False  # Prevent SHAP from auto-displaying the plot
        )

        st.pyplot(fig)

        # Display the feature contributions in a table
        st.write("### Feature Contributions:")
        st.table(st.session_state.feature_contributions)




