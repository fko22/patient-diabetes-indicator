import os
import streamlit as st
import pandas as pd
import shap
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def explain_model(feature_names, X_sample, X_sample_scaled, rf_model):
    """
    Explain the model's prediction for a single sample using SHAP values.

    Parameters:
    - feature_names: List of feature names.
    - X_sample: Original input data for the sample.
    - X_sample_scaled: Scaled input data for the sample.
    - rf_model: Trained Random Forest model.

    Returns:
    - shap_values: SHAP values for the sample.
    - feature_contributions: DataFrame with feature names and SHAP values.
    """
    # Initialize SHAP Tree Explainer
    explainer = shap.TreeExplainer(rf_model)

    # Compute SHAP values for the scaled sample
    shap_values = explainer.shap_values(X_sample_scaled)

    # Extract SHAP values for class 1 (positive class)
    # shap_values_class_1 = shap_values[1][0]  # SHAP values for class 1, first sample
    shap_values_class_1 = shap_values[0][:, 1]  # SHAP values for class 1, all features

    # Ensure SHAP values and feature names align
    assert len(feature_names) == len(shap_values_class_1), "Mismatch in feature and SHAP value lengths!"

    # Create a DataFrame for feature contributions
    feature_contributions = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values_class_1
    })

    # Sort features by absolute SHAP value
    feature_contributions = feature_contributions.reindex(
        feature_contributions["SHAP Value"].abs().sort_values(ascending=False).index
    )

    return shap_values, feature_contributions


def load_api_key():
    # Load environment variables from .env for local testing
    load_dotenv()

    api_key = None

    # Check if API key is already set
    if not os.environ.get("OPENAI_API_KEY"):
        try:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        except Exception:
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Use the API key
    api_key = os.environ["OPENAI_API_KEY"]
    print("api_key",api_key)

    if not api_key:
        raise ValueError("API key not found. Please set it in .env or Streamlit secrets.")

    # Test the API key with ChatOpenAI
    chat = ChatOpenAI(api_key=api_key)

    # return chat and whether the API key is valid
    return chat, api_key


def generate_recommendations(feature_contributions, feature_info, user_model_text, llm):
    """
    Generate personalized lifestyle recommendations based on SHAP values.

    Parameters:
    - feature_contributions: DataFrame with features and SHAP values
    - feature_info: Dictionary with feature descriptions
    - prediction_result: String ("Diabetes Present" or "No Diabetes")
    - diabetes_prob: Float, probability of the predicted class

    Returns:
    - Recommendations from the LLM
    """
    # Identify top positive and negative contributing features
    positive_features = feature_contributions[feature_contributions["SHAP Value"] > 0]
    negative_features = feature_contributions[feature_contributions["SHAP Value"] < 0]

    # Create prompt for LLM
    prompt = f"""
    A patient's health indicators have been analyzed using SHAP (Shapley) values for diabetes risk prediction.

    {user_model_text}

    Here are the contributing factors:

    - Features contributing to increased diabetes risk:
    {positive_features.to_string(index=False, header=False)}

    - Features contributing to reduced diabetes risk:
    {negative_features.to_string(index=False, header=False)}

    Provide personalized lifestyle recommendations for this patient.
    Focus on:
    - Areas where they should improve their habits (based on positive SHAP values)
    - Areas where they are doing well (based on negative SHAP values)

    Respond as if you're talking to the patient directly. Avoid mentioning SHAP values or technical details.

    First mention the result and probability of the prediction (underlined and bolded). Then, provide clear and actionable recommendations.

    Don't start with "Dear patient" or similar formalities. Don't mention the feature names (e.g., "HighBP"), mention them in full ("High blood pressure").

    Break down your answer into bullet points for clarity. Make it easy to understand and actionable.
    """

    # Use the ChatOpenAI model to generate recommendations
    response = llm(prompt)
    return response

