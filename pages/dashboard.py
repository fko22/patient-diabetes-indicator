import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.ticker as ticker
from datetime import datetime
from email.mime.text import MIMEText
import smtplib

if not st.session_state.logged_in:
    st.warning("You need to log in first")
else:
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
        features = [col for col in user_data.columns if col not in ['user_id', 'date', 'email', 'Sex', 'Age', 'Education', 'id', 'name', 'Probability']]

        default_selection = ['Prediction'] + features[:4]  # Include 'Prediction' first and then the next 4 features

        selected_features = st.multiselect(
            "Select features to display on the graph:",
            features,
            default=default_selection
        )

        print("features: ", features)

        # Display separate graphs for selected features
        for feature in selected_features:
            st.subheader(f"{feature}")

            user_data['date'] = pd.to_datetime(user_data['date']).dt.date
            print(user_data['date'])
            
            if feature == "BMI":
                fig, ax = plt.subplots(figsize=(10, 6))

                # Define the healthy BMI range
                healthy_bmi_mask = (user_data[feature] >= 18.5) & (user_data[feature] <= 25)
                user_data['is_healthy'] = healthy_bmi_mask

                # Prepare the data
                dates = user_data['date'].values
                values = user_data[feature].values

                # Convert dates to numeric for LineCollection
                numeric_dates = mdates.date2num(dates)
                points = np.array([numeric_dates, values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Define the colors for healthy and not healthy BMI
                colors = ['green' if is_healthy else 'red' for is_healthy in user_data['is_healthy']]

                # Create a LineCollection with the segments and colors
                lc = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(lc)

                for x, y, color in zip(dates, values, colors):
                    ax.scatter(x, y, color=color, zorder=5)

                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

                # Set axis labels and title
                ax.set_xlim(numeric_dates.min(), numeric_dates.max())
                ax.set_ylim(values.min() - 1, values.max() + 1)
                ax.set_title(f"{feature} Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("BMI")
                legend_elements = [
                    Line2D([0], [0], color='green', lw=2, label='BMI (Healthy)'),
                    Line2D([0], [0], color='red', lw=2, label='BMI (Not Healthy)')
                ]
                ax.legend(handles=legend_elements, loc='upper left')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

            elif feature in ["MentHlth", "PhysHlth"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(user_data['date'], user_data[feature], label=feature, marker='o', linestyle='-')
                ax.set_title(f"{feature} Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Number of unhealthy days")
                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                plt.xticks(rotation=45, ha='right')
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
                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                ax.set_xlabel("Date")
                ax.set_ylabel("Income Level")
                
                plt.xticks(rotation=45, ha='right')  

                ax.set_yticks(range(1, 9))
                ax.set_yticklabels(income_map.values())
                ax.legend()
                st.pyplot(fig)

            elif feature in ["Prediction","Probability"]:
                fig, ax = plt.subplots(figsize=(10, 6))

                dates = user_data['date']
                numeric_dates = mdates.date2num(dates)
                probabilities = user_data['Probability'] * 100  # Scale probability to percentage

                # Combine the prediction and probability into segments
                points = np.array([numeric_dates, probabilities]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Define colors based on predictions
                colors = ['green' if pred == "No Diabetes Present" else 'red' for pred in user_data['Prediction']]

                # Create a LineCollection with segments and colors
                lc = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(lc)

                for x, y, color in zip(dates, probabilities, colors):
                    ax.scatter(x, y, color=color, zorder=5)

                # Set axis limits and labels
                ax.set_xlim(numeric_dates.min(), numeric_dates.max())
                ax.set_ylim(probabilities.min() - 10, probabilities.max() + 10)
                ax.set_title("Your Risk of Diabetes Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Risk of Diabetes (%)")

                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                plt.xticks(rotation=45, ha='right')

                # Add a legend
                legend_elements = [
                    Line2D([0], [0], color='green', lw=2, label='No Diabetes Present'),
                    Line2D([0], [0], color='red', lw=2, label='Diabetes Present')
                ]
                ax.legend(handles=legend_elements, loc='upper left')

                # Show the plot
                st.pyplot(fig)
                
            elif feature in ["CholCheck","PhysActivity","Fruits","Veggies","AnyHealthcare"]:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Convert dates to numeric for LineCollection
                dates = user_data['date']
                numeric_dates = mdates.date2num(dates)
                binary_values = user_data[feature].astype(int)  # Convert Yes/No to 1/0

                # Prepare segments for LineCollection
                points = np.array([numeric_dates, binary_values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Define colors based on binary values
                colors = ['green' if val == 1 else 'red' for val in binary_values]

                # Create LineCollection
                lc = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(lc)

                for x, y, color in zip(dates, binary_values, colors):
                    ax.scatter(x, y, color=color, zorder=5)

                # Set axis limits and labels
                ax.set_xlim(numeric_dates.min(), numeric_dates.max())
                ax.set_ylim(-0.5, 1.5)  # Binary range
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["No", "Yes"])
                ax.set_title(f"{feature} Distribution Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Yes/No")

                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                plt.xticks(rotation=45, ha='right')

                # Add legend
                legend_elements = [
                    Line2D([0], [0], color='green', lw=2, label='Yes'),
                    Line2D([0], [0], color='red', lw=2, label='No')
                ]
                ax.legend(handles=legend_elements, loc='upper left')

                # Show the plot
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Convert dates to numeric for LineCollection
                dates = user_data['date']
                numeric_dates = mdates.date2num(dates)
                binary_values = user_data[feature].astype(int)  # Convert Yes/No to 1/0

                # Prepare segments for LineCollection
                points = np.array([numeric_dates, binary_values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Define colors based on binary values
                colors = ['red' if val == 1 else 'green' for val in binary_values]

                # Create LineCollection
                lc = LineCollection(segments, colors=colors, linewidth=2)
                ax.add_collection(lc)

                for x, y, color in zip(dates, binary_values, colors):
                    ax.scatter(x, y, color=color, zorder=5)

                # Set axis limits and labels
                ax.set_xlim(numeric_dates.min(), numeric_dates.max())
                ax.set_ylim(-0.5, 1.5)  # Binary range
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["No", "Yes"])
                ax.set_title(f"{feature} Distribution Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Yes/No")

                # Format x-axis for dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                if len(dates) <= 10:
                    ax.set_xticks(dates)  # Set ticks for all available dates
                else:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                plt.xticks(rotation=45, ha='right')

                # Add legend
                legend_elements = [
                    Line2D([0], [0], color='red', lw=2, label='Yes'),
                    Line2D([0], [0], color='green', lw=2, label='No')
                ]
                ax.legend(handles=legend_elements, loc='upper left')

                # Show the plot
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
                    msg['From'] = "healthtrackdiabetes@gmail.com" 
                    msg['To'] = email_address

                    with smtplib.SMTP("smtp.gmail.com", 587) as server: 
                        server.starttls()
                        server.login("healthtrackdiabetes@gmail.com", "ycajycbjfhsuvdkb") 
                        server.sendmail("healthtrackdiabetes@gmail.com", email_address, msg.as_string())
                    
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
            else:
                st.warning("Please enter a valid email address.")

    if __name__ == "__main__":
        dashboard_page()