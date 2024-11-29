import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime
from email.mime.text import MIMEText
import smtplib

# Connect to SQLite database
conn = sqlite3.connect('user_predictions.db')

# Function to fetch users
def fetch_users():
    query = '''
    SELECT DISTINCT 
        predictions.user_id, 
        users.email, 
        predictions.Sex, 
        predictions.Age, 
        predictions.Education
    FROM predictions
    INNER JOIN users
    ON users.unique_id = predictions.user_id
    '''
    return pd.read_sql_query(query, conn)

# Function to fetch data for a specific user
def fetch_user_data(user_id):
    query = f'''
    SELECT 
        predictions.*, 
        users.email,
        users.name
    FROM predictions
    INNER JOIN users
    ON users.unique_id = predictions.user_id
    WHERE predictions.user_id = '{user_id}'
    ORDER BY date DESC
    '''
    return pd.read_sql_query(query, conn)

def map_value(value, mapping, default="Unknown"):
    return mapping.get(value, default)

education_options = {
    1: "Never attended school or only kindergarten",
    2: "Grades 1 through 8 (Elementary)",
    3: "Grades 9 through 11 (Some high school)",
    4: "Grade 12 or GED (High school graduate)",
    5: "College 1 year to 3 years (Some college or technical school)",
    6: "College 4 years or more (College graduate)"
}

age_range_map = {
    1: "0-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+"
}


# Dashboard page
def dashboard_page():
    st.title("Dashboard")
    
    # Select user by ID or email
    users_df = fetch_users()
    user_selection = st.selectbox(
        "Select a user",
        users_df.apply(lambda x: f"{x['user_id']} - {x['email']}", axis=1)
    )
    selected_user_id = user_selection.split(' - ')[0]
    
    # Fetch user data
    user_data = fetch_user_data(selected_user_id)
    if user_data.empty:
        st.warning("No data available for this user.")
        return
    
    conn.close() 
    
    # Display basic information
    st.subheader("User Information")
    user_info = user_data.iloc[0]
    education_desc = map_value(user_info['Education'], education_options)
    age_range = age_range_map.get(user_info["Age"], "Unknown")
    box_style = """
    <style>
        .tile-box {
            border: 2px solid #4CAF50;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
            background-color: #f9f9f9;
            text-align: center;
        }
        .metric-box {
            display: inline-block;
            width: 100%;
            text-align: center;
            font-size: 20px;
            color: black;
        }
    </style>
"""

    st.markdown(box_style, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="tile-box"><div class="metric-box"><b>Name:</b> ' + user_info["name"] + '</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tile-box"><div class="metric-box"><b>Email:</b> ' + user_info["email"] + '</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="tile-box"><div class="metric-box"><b>Age:</b> ' + age_range + '</div></div>', unsafe_allow_html=True)

    col4, col5 = st.columns(2)
    with col4:
        gender = 'Male' if user_info['Sex'] == 1 else 'Female'
        st.markdown('<div class="tile-box"><div class="metric-box"><b>Gender:</b> ' + gender + '</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="tile-box"><div class="metric-box"><b>Education Level:</b> ' + education_desc + '</div></div>', unsafe_allow_html=True)
        
    st.subheader("Feature Selection")
    features = [col for col in user_data.columns if col not in ['user_id', 'date', 'email', 'Sex', 'Age', 'Education']]
    selected_features = st.multiselect(
            "Select features to display on the graph:",
            features,
            default=features[:5] 
        )
        
    # Display separate graphs for selected features
    for feature in selected_features:
        st.subheader(f"{feature}")

        user_data['date'] = pd.to_datetime(user_data['date'])
        
        if feature == "BMI":
            fig, ax = plt.subplots(figsize=(10, 6))
            healthy_bmi_mask = (user_data[feature] >= 18.5) & (user_data[feature] <= 25)
    
            ax.plot(user_data['date'][healthy_bmi_mask], user_data[feature][healthy_bmi_mask], label=f"{feature} (Healthy)", marker='o', linestyle='-', color='green')
            ax.plot(user_data['date'][~healthy_bmi_mask], user_data[feature][~healthy_bmi_mask], label=f"{feature} (Not Healthy)", marker='o', linestyle='-', color='red')
            
            ax.set_title(f"{feature} Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("BMI")
            
            plt.xticks(rotation=45, ha='right')
            
            ax.legend()
            st.pyplot(fig)

        elif feature in ["MentHlth", "PhysHlth"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(user_data['date'], user_data[feature], label=feature, marker='o', linestyle='-')
            ax.set_title(f"{feature} Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of unhealthy days")
            plt.xticks(rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
        
        elif feature == 'Income':
            income_map = {
                1: "Less than $10,000", 2: "$10,000 - $14,999", 3: "$15,000 - $19,999",
                4: "$20,000 - $24,999", 5: "$25,000 - $34,999", 6: "$35,000 - $49,999",
                7: "$50,000 - $74,999", 8: "$75,000 and above"
            }
            user_data['IncomeCategory'] = user_data['Income'].map(income_map)

            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(user_data['date'], user_data['Income'], label='Income', marker='o', linestyle='-', color='b')
            
            ax.set_title(f"{feature} Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Income Level")
            
            plt.xticks(rotation=45, ha='right')  

            ax.set_yticks(range(1, 9))
            ax.set_yticklabels(income_map.values())
            ax.legend()
            st.pyplot(fig)

        elif feature in ["Prediction","Probability"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            green_mask = user_data['Prediction'] == "No Diabetes Present"
            red_mask = user_data['Prediction'] == "Diabetes Present"

            ax.plot(user_data['date'][green_mask], user_data['Probability'][green_mask] * 100, 
                    label="No Diabetes Present", marker='o', linestyle='-', color='green')

            ax.plot(user_data['date'][red_mask], -user_data['Probability'][red_mask] * 100, 
                    label="Diabetes Present", marker='o', linestyle='-', color='red')

            ax.set_title("Prediction vs Probability Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Probability (%)")
            
            plt.xticks(rotation=45, ha='right')

            ax.set_ylim(bottom=-100, top=100)  
            ax.legend()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))

            binary_counts = user_data.groupby(['date', feature]).size().unstack().fillna(0)
            binary_counts.plot(kind='line', ax=ax, marker='o', color=["green", "red"])

            ax.set_title(f"{feature} Distribution Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Yes/No")

            ax.set_ylim(-0.5, 1.5)  

            ax.set_yticks([1, 0])  
            ax.set_yticklabels(["Yes", "No"])  

            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

    
    st.subheader("Email Dashboard")
    email_address = st.text_input("Enter email address:")
    if st.button("Send Email"):
        if email_address:
            # Prepare email content
            content = f"Dashboard for user {selected_user_id}\n\n"
            content += user_data.to_string(index=False)
            
            try:
                msg = MIMEText(content)
                msg['Subject'] = "Your Dashboard Results"
                msg['From'] = "youremail@example.com"  # replace with your email
                msg['To'] = email_address

                with smtplib.SMTP("smtp.example.com", 587) as server:  # need to replace with actual SMTP server
                    server.starttls()
                    server.login("youremail@example.com", "yourpassword")  # need to replace with credentials
                    server.sendmail("youremail@example.com", email_address, msg.as_string())
                
                st.success("Email sent successfully!")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
        else:
            st.warning("Please enter a valid email address.")

if __name__ == "__main__":
    dashboard_page()