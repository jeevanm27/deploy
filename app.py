import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import joblib
import tensorflow as tf
import mysql.connector
from mysql.connector import Error
import hashlib
import re
import datetime
import uuid
from datetime import datetime, timedelta

# --- Import Google Cloud Translation Client ---
from google.cloud import translate_v2 as translate

# --- Session State Initialization ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'model_loading_status' not in st.session_state:
    st.session_state.model_loading_status = {
        'pest_model': False,
        'disease_model': False,
        'crop_model': False,
        'fertilizer_model': False
    }

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'agrizen_db',
    'user': 'root',
    'password': 'your_password',  # Change to your actual MySQL password
    'port': 3306
}

# --- Database Connection Function ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except Error as e:
        st.error(f"Database Error: {e}")
        return None

# --- Save Feedback Function ---
# Ensure your database has a table named 'feedback' with columns such as:
# id (AUTO_INCREMENT), user_id, page, feedback_text, created_at
def save_feedback(page, feedback_text):
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            user_id = st.session_state.user_id if st.session_state.is_authenticated else None
            created_at = datetime.now()
            feedback_query = """
                INSERT INTO feedback (user_id, page, feedback_text, created_at)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(feedback_query, (user_id, page, feedback_text, created_at))
            conn.commit()
        except Error as e:
            st.error(f"Error saving feedback: {e}")
        finally:
            cursor.close()
            conn.close()

# --- Feedback Widget Renderer ---
def render_feedback_widget(page):
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("Give Feedback", expanded=False):
        feedback_text = st.text_area("Enter your feedback", key=f"feedback_{page}")
        if st.button("Submit Feedback", key=f"submit_feedback_{page}"):
            if not feedback_text.strip():
                st.error("Please enter feedback before submitting.")
            else:
                save_feedback(page, feedback_text)
                st.success("Thank you for your feedback!")

# --- Google Translate Function ---
def translate_text(text, target_language):
    try:
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language=target_language)
        return result['translatedText']
    except Exception as e:
        st.error("Translation error: " + str(e))
        return text

# --- Password Hashing Function ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Email Validation Function ---
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

# --- Set Page Configuration ---
st.set_page_config(page_title="AgriZen Dashboard", page_icon="🌾", layout="wide")
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

# --- CSS Styling ---
st.markdown(
    """
    <style>
    /* Set the background image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Logo and navbar styles */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: rgba(0, 0, 0, 0.5);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }
    .logo {
        font-size: 28px;
        font-weight: bold;
        color: white;
        text-decoration: none;
    }
    .auth-buttons {
        display: flex;
        gap: 15px;
    }
    .auth-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .auth-button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .logout-button {
        background-color: #f44336;
    }
    .logout-button:hover {
        background-color: #d32f2f;
    }
    
    /* Main container padding for navbar */
    .main-container {
        margin-top: 80px;
        padding: 20px;
    }
    
    /* Authentication forms */
    .auth-form {
        max-width: 400px;
        margin: 0 auto;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .auth-form h2 {
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    .form-input {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
    }
    .form-button {
        width: 100%;
        padding: 12px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        margin-top: 15px;
    }
    .form-button:hover {
        background-color: #45a049;
    }
    .form-switch {
        text-align: center;
        margin-top: 20px;
        color: #333;
    }
    .form-switch a {
        color: #4CAF50;
        text-decoration: none;
        font-weight: bold;
    }
    
    /* Page heading */
    h1, h2, h3 {
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Card container using CSS Grid for a 3x2 layout */
    .card-container {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 40px;
      justify-content: center;
      width: 100%;
      max-width: 1200px;
      margin: 40px auto;
    }
    .card {
      background-color: rgba(255, 255, 255, 0.25);
      border-radius: 10px;
      height: 300px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.3);
      text-align: center;
      cursor: pointer;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      color: white;
      padding: 0;
    }
    .card:hover {
      transform: scale(1.03);
      box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .card img {
      width: 100%;
      height: 70%;
      object-fit: cover;
      margin: 0;
      padding: 0;
    }
    .card .card-title {
      font-size: 20px;
      font-weight: bold;
      margin: 15px;
    }
    
    /* Back button styling */
    .back-button {
      display: inline-block;
      background-color: rgba(255, 255, 255, 0.75);
      color: #000000;
      padding: 10px 20px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 16px;
      transition: all 0.3s ease;
      margin-bottom: 20px;
    }
    .back-button:hover {
      background-color: rgba(255, 255, 255, 1);
      transform: translateX(-5px);
    }
    
    /* User welcome message */
    .user-welcome {
        color: white;
        margin-bottom: 20px;
        font-size: 18px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
    }
    
    /* Make form elements look better in dark mode */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stMultiselect {
        color: #333;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Navigation Bar ---
def render_navbar():
    if st.session_state.is_authenticated:
        navbar_html = f"""
        <div class="navbar">
            <a href="?page=Home" class="logo">AGRI AI</a>
            <div class="auth-buttons">
                <span style="color: white; margin-right: 15px;">Welcome, {st.session_state.username}</span>
                <a href="?page=Dashboard" class="auth-button">Dashboard</a>
                <a href="?action=logout" class="auth-button logout-button">Logout</a>
            </div>
        </div>
        """
    else:
        navbar_html = """
        <div class="navbar">
            <a href="?page=Home" class="logo">AGRI AI</a>
            <div class="auth-buttons">
                <a href="?page=Login" class="auth-button">Login</a>
                <a href="?page=Register" class="auth-button">Register</a>
            </div>
        </div>
        """
    st.markdown(navbar_html, unsafe_allow_html=True)

# --- Render Back Button ---
def render_back_button():
    st.markdown(
        """
        <a href="?page=Dashboard" class="back-button">← Back to Dashboard</a>
        """,
        unsafe_allow_html=True,
    )

# --- Processing Logout ---
query_params = st.query_params
if "action" in query_params and query_params["action"][0] == "logout":
    st.session_state.is_authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.query_params.clear()
    st.rerun()

# --- Render Navbar ---
render_navbar()

# --- Main Container ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Navigation via Query Parameters ---
if "page" in query_params:
    current_page = query_params["page"][0]
else:
    current_page = "Home"

# --- Remedy Mappings for Diseases ---
disease_remedies = {
    "Apple_Apple_scab": """
Apple Scab Remedies:

Crop Management:
- Remove infected leaves and fruits.
- Prune trees to improve air circulation.
- Apply fungicides during spring.
- Use resistant cultivars.

Organic Remedies:
- Neem Oil spray.
- Baking Soda spray.
- Garlic and Onion extract.

Inorganic Remedies:
- Copper-based fungicides.
- Chlorothalonil.
""",
    # ... other disease remedy mappings ...
    "Tomato_Tomato_mosaic_virus": """
Tomato Mosaic Virus Remedies:

Crop Management:
- Remove infected plants.
- Use resistant varieties.
- Sanitize tools.

Organic Remedies:
- Control aphids with neem oil or insecticidal soap.
- Practice crop rotation.

Inorganic Remedies:
- Insecticides to control vectors.
"""
}

# --- Remedy Mappings for Pests ---
pest_remedies = {
    "aphids": """
Aphids Remedies:

Crop Recommendations:
- Cabbage, Tomato, Chili, Cucumber, Beans, Peas, Peppers.

Organic Remedies:
- Neem Oil: Repels and disrupts aphid feeding.
- Garlic and Pepper Spray: Homemade mix to deter aphids.
- Introduce Natural Predators: Ladybugs, lacewing larvae, parasitic wasps.

Inorganic Remedies:
- Insecticidal Soap.
- Imidacloprid.
""",
    # ... other pest remedy mappings ...
    "stem_borer": """
Stem Borer Remedies:

Crop Recommendations:
- Rice, Sugarcane, Corn, Maize.

Organic Remedies:
- Neem Oil: Inhibits larvae development.
- Trichogramma Wasps.
- Bt-based Sprays.

Inorganic Remedies:
- Chlorpyrifos.
- Endosulfan.
"""
}

# --- Home Page (Landing Page) ---
if current_page == "Home":
    st.title("Welcome to Agricultural AI Platform")
    st.write("Empowering farmers with modern technology for sustainable and efficient farming")
    
    st.markdown("""
    <div style="text-align: center; margin: 50px auto; max-width: 800px; background-color: rgba(0,0,0,0.6); padding: 30px; border-radius: 15px;">
        <h2>Revolutionize Your Farming Practices</h2>
        <p style="color: white; font-size: 18px; margin: 20px 0;">
            Our AI-powered platform helps farmers identify plant diseases, detect pests, recommend crops,
            optimize fertilizer usage, and manage irrigation effectively.
        </p>
        <div style="margin-top: 30px;">
            <a href="?page=Register" class="auth-button" style="font-size: 18px; padding: 12px 24px; margin: 10px;">Get Started</a>
            <a href="?page=Login" class="auth-button" style="font-size: 18px; padding: 12px 24px; margin: 10px;">Login</a>
        </div>
    </div>
    
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin-top: 50px;">
        <div style="width: 300px; text-align: center; margin: 20px; background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px;">
            <h3>Smart Disease Detection</h3>
            <p style="color: white;">Identify plant diseases early with our AI-powered image recognition technology</p>
        </div>
        <div style="width: 300px; text-align: center; margin: 20px; background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px;">
            <h3>Pest Identification</h3>
            <p style="color: white;">Quickly identify pests and get targeted remedies to protect your crops</p>
        </div>
        <div style="width: 300px; text-align: center; margin: 20px; background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px;">
            <h3>Custom Recommendations</h3>
            <p style="color: white;">Get personalized crop and fertilizer recommendations based on your soil conditions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Login Page ---
elif current_page == "Login":
    st.markdown(
        """
        <div class="auth-form">
            <h2>Login to Your Account</h2>
        """,
        unsafe_allow_html=True,
    )
    
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    login_button = st.button("Login")
    
    if login_button:
        if not email or not password:
            st.error("Please fill in all fields")
        else:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor(dictionary=True)
                    hashed_password = hash_password(password)
                    query = "SELECT * FROM users WHERE email = %s AND password = %s"
                    cursor.execute(query, (email, hashed_password))
                    user = cursor.fetchone()
                    
                    if user:
                        st.session_state.is_authenticated = True
                        st.session_state.user_id = user['id']
                        st.session_state.username = user['username']
                        st.success("Login successful!")
                        st.experimental_set_query_params(page="Dashboard")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid email or password.")
                except Error as e:
                    st.error(f"Error: {e}")
                finally:
                    cursor.close()
                    conn.close()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Register Page ---
elif current_page == "Register":
    st.markdown(
        """
        <div class="auth-form">
            <h2>Create a New Account</h2>
        """,
        unsafe_allow_html=True,
    )
    username = st.text_input("Username", key="register_username")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
    
    register_button = st.button("Register")
    
    if register_button:
        if not username or not email or not password or not confirm_password:
            st.error("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        elif not is_valid_email(email):
            st.error("Invalid email format")
        else:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    hashed_password = hash_password(password)
                    insert_query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
                    cursor.execute(insert_query, (username, email, hashed_password))
                    conn.commit()
                    st.success("Registration successful! Please login.")
                    st.experimental_set_query_params(page="Login")
                    st.experimental_rerun()
                except Error as e:
                    st.error(f"Database Error: {e}")
                finally:
                    cursor.close()
                    conn.close()

    st.markdown(
        """
        <div class="form-switch">
            Already have an account? <a href="?page=Login">Login here</a>.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Dashboard Page ---
elif current_page == "Dashboard":
    if not st.session_state.is_authenticated:
        st.error("You must be logged in to view the dashboard.")
    else:
        st.title("Dashboard")
        st.write(f"Welcome, {st.session_state.username}!")
        
        st.markdown("""
        <div class="card-container">
            <div class="card" onclick="window.location.href='?page=DiseaseDetection'">
                <img src="https://images.unsplash.com/photo-1568605114967-8130f3a36994?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60" alt="Disease Detection">
                <div class="card-title">Disease Detection</div>
            </div>
            <div class="card" onclick="window.location.href='?page=PestIdentification'">
                <img src="https://images.unsplash.com/photo-1591100859400-c9e6e2ad3cc2?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60" alt="Pest Identification">
                <div class="card-title">Pest Identification</div>
            </div>
            <div class="card" onclick="window.location.href='?page=CropRecommendation'">
                <img src="https://images.unsplash.com/photo-1600267168478-5b171ea6c9f0?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60" alt="Crop Recommendation">
                <div class="card-title">Crop Recommendation</div>
            </div>
            <div class="card" onclick="window.location.href='?page=FertilizerOptimization'">
                <img src="https://images.unsplash.com/photo-1587390964356-95fa10d0b2f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60" alt="Fertilizer Optimization">
                <div class="card-title">Fertilizer Optimization</div>
            </div>
            <div class="card" onclick="window.location.href='?page=IrrigationManagement'">
                <img src="https://images.unsplash.com/photo-1519203446144-5a8b96aa3a75?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60" alt="Irrigation Management">
                <div class="card-title">Irrigation Management</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Disease Detection Page (Example Model Page) ---
elif current_page == "DiseaseDetection":
    if not st.session_state.is_authenticated:
        st.error("Please login to access this feature.")
    else:
        st.title("Disease Detection")
        st.write("Upload an image of the affected plant to detect the disease.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            # Placeholder for model prediction
            st.write("Processing image and predicting disease...")
            # Example prediction
            predicted_disease = "Tomato_Late_blight"
            st.write(f"Predicted Disease: {predicted_disease}")
            if predicted_disease in disease_remedies:
                st.markdown(disease_remedies[predicted_disease])
                # --- Translation Widget for Remedies ---
                with st.expander("Translate Remedies to Local Language"):
                    target_language = st.selectbox("Select Language", 
                                                   options=["hi", "ta", "te", "bn", "mr"], 
                                                   key="translate_lang")
                    if st.button("Translate", key="translate_btn"):
                        translated_remedy = translate_text(disease_remedies[predicted_disease], target_language)
                        st.markdown(translated_remedy, unsafe_allow_html=True)
            else:
                st.write("No remedies found for the predicted disease.")
        render_back_button()
        # --- Render Feedback Widget for Disease Detection page ---
        render_feedback_widget("DiseaseDetection")

# --- Additional Pages (e.g., PestIdentification, CropRecommendation, etc.) ---
# For each additional model page, add your content and then call:
# render_feedback_widget("<PageName>")
# and optionally add a similar translation widget for remedy texts.

# --- Close Main Container ---
st.markdown("</div>", unsafe_allow_html=True)