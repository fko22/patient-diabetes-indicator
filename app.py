import streamlit as st
import sqlite3

# Database setup
def init_db():
    conn = sqlite3.connect('user_predictions.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        unique_id TEXT UNIQUE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unique_id TEXT,
        date TEXT,
        inputs TEXT,
        prediction_result TEXT,
        diabetes_probability REAL,
        no_diabetes_probability REAL,
        FOREIGN KEY (unique_id) REFERENCES users (unique_id)
    )
    """)
    conn.commit()
    conn.close()

# Generate unique ID
def generate_unique_id(name):
    conn = sqlite3.connect('user_predictions.db')
    cursor = conn.cursor()
    
    last_name = name.split()[-1].lower()
    cursor.execute("SELECT unique_id FROM users WHERE unique_id LIKE ?", (f"{last_name}%",))
    existing_ids = cursor.fetchall()
    
    if not existing_ids:
        return f"{last_name}1"
    
    existing_numbers = [int(id[0].replace(last_name, "")) for id in existing_ids if id[0].replace(last_name, "").isdigit()]
    next_number = max(existing_numbers, default=0) + 1
    return f"{last_name}{next_number}"

# Save user details
def save_user(name, email, unique_id):
    conn = sqlite3.connect('user_predictions.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (name, email, unique_id) VALUES (?, ?, ?)", (name, email, unique_id))
    conn.commit()
    conn.close()

# Check if user exists
def user_exists(unique_id=None, email=None):
    conn = sqlite3.connect('user_predictions.db')
    cursor = conn.cursor()
    if unique_id:
        cursor.execute("SELECT name FROM users WHERE unique_id = ?", (unique_id,))
    elif email:
        cursor.execute("SELECT name FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    return result

# Initialize database on startup
init_db()

# Track login status in session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None

# Login page
def login_page():
    st.header("Login")

    option = st.radio("How would you like to log in?", ["Name and Email", "Unique ID"])
    if option == "Name and Email":
        name = st.text_input("Enter your name")
        email = st.text_input("Enter your email")
        if st.button("Submit"):
            if name and email:
                user = user_exists(email=email)
                if user:
                    conn = sqlite3.connect('user_predictions.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT unique_id FROM users WHERE email = ?", (email,))
                    existing_id = cursor.fetchone()[0]
                    conn.close()
                    st.session_state.logged_in = True
                    st.session_state.user_id = existing_id
                    st.success(f"Welcome back, {user[0]}!")
                else:
                    unique_id = generate_unique_id(name)
                    save_user(name, email, unique_id)
                    st.session_state.logged_in = True
                    st.session_state.user_id = unique_id
                    st.success(f"Welcome, {name}! Your unique ID is **{unique_id}**.")
            else:
                st.error("Please provide both name and email.")
    elif option == "Unique ID":
        unique_id = st.text_input("Enter your unique ID")
        if st.button("Submit"):
            if unique_id:
                user = user_exists(unique_id=unique_id)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = unique_id
                    st.success(f"Welcome back, {user[0]}!")
                else:
                    st.error("Unique ID not found. Please log in with your name and email.")
            else:
                st.error("Please provide your unique ID.")

# Navigation and page control
def render_pages():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Predictions"])

    if page == "Home":
        st.header("Welcome to the Diabetes Prediction App!")
        st.write(f"Logged in as User ID: {st.session_state.user_id}")
    elif page == "Predictions" and st.session_state.logged_in:
        st.header("Prediction Page")
        st.write("This is the predictions page.")

# App logic
if not st.session_state.logged_in:
    st.warning("You must log in to access the app.")
    login_page()
else:
    render_pages()
