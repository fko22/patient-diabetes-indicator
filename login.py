import streamlit as st
import sqlite3

st.set_page_config(page_title="Login", page_icon="🔑")
st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
    }
    </style>
    <h1 class="center-text">🌿 HealthTrack Diabetes 🌿</h1>
    <h3 class="center-text" style="color: white;">Predicting Diabetes Risk and Providing Lifestyle Recommendations</h3>
    """,
    unsafe_allow_html=True
)
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
def user_exists(unique_id=None, email=None, name=None):
    conn = sqlite3.connect('user_predictions.db')
    cursor = conn.cursor()
    if unique_id:
        cursor.execute("SELECT name FROM users WHERE unique_id = ?", (unique_id,))
    elif email:
        cursor.execute("SELECT name FROM users WHERE email = ? and name=?", (email,name,))
    result = cursor.fetchone()
    conn.close()
    return result

# Initialize database on startup
init_db()

# Track login status in session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.log_in_method = None
    st.session_state.user = None
    st.session_state.name = None

# Login page
def login_page():
    st.header("Login")
    if not st.session_state.logged_in:
        st.warning("You must log in to access the app")

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
                        st.session_state.log_in_method = "existing_user"
                        st.session_state.user = user[0]
                    else:
                        unique_id = generate_unique_id(name)
                        save_user(name, email, unique_id)
                        st.session_state.logged_in = True
                        st.session_state.user_id = unique_id
                        st.session_state.log_in_method = "new_user"
                    st.session_state.name = name
                    st.rerun()
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
                        st.session_state.log_in_method = "existing_user"
                        st.session_state.user = user[0]
                        st.rerun()
                    else:
                        st.error("Unique ID not found. Please log in with your name and email.")
                else:
                    st.error("Please provide your unique ID.")
    else:
        if st.session_state.log_in_method == "existing_user":
            st.success(f"Welcome back, {st.session_state.user}!")
        elif st.session_state.log_in_method == "new_user":
                st.success(f"Welcome, {st.session_state.name }! Your unique ID is **{st.session_state.user_id}**.")


login_page()

