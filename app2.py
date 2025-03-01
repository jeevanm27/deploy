import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

# --- Set Page Config ---
st.set_page_config(page_title="AgriZen Dashboard", page_icon="🌾", layout="wide")

# --- CSS Styling ---
st.markdown(
    """
    <style>
    /* Background image for the entire page */
    body {
        background-image: url('https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    /* Logo container positioned fixed in the top-left */
    .logo-container {
        position: fixed;
        top: 10px;
        left: 10px;
        display: flex;
        align-items: center;
        z-index: 1000;
    }
    .logo-container img {
        height: 50px;
        margin-right: 10px;
    }
    .logo-container h1 {
        font-size: 24px;
        color: white;
        margin: 0;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    /* Card container using CSS Grid for a 3x2 layout */
    .card-container {
      display: grid;
      grid-template-columns: repeat(3, 400px);
      gap: 20px;
      justify-content: center;
      margin-top: 100px; /* extra margin to account for the fixed logo */
      width: 100%;
    }
    .card {
      background-color: rgba(255, 255, 255, 0.25);
      border-radius: 10px;
      width: 400px;
      height: 300px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.3);
      text-align: center;
      cursor: pointer;
      transition: transform 0.2s;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      color: white;
      padding: 0;
    }
    .card:hover {
      transform: scale(1.05);
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
      margin: 10px;
    }
    /* Back button styling */
    .back-button {
      display: inline-block;
      background-color: rgba(255, 255, 255, 0.75);
      color: white;
      padding: 10px 20px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 16px;
    }
    .back-button:hover {
      background-color: rgba(255, 255, 255, 1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Logo in Top-Left ---
st.markdown(
    """
    <div class="logo-container">
        <img src="https://i.pinimg.com/736x/e5/7b/97/e57b975cb18063bfc3a84cfcf5dcc863.jpg" alt="Logo">
        <h1>AGRI AI</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Navigation via Query Parameters ---
# Use st.query_params (a property) to decide which page to display
query_params = st.query_params
if "page" in query_params:
    current_page = query_params["page"][0]
else:
    current_page = "Home"

# Helper function to render a back button
def render_back_button():
    st.markdown("<a href='?page=Home' target='_self' class='back-button'>Back to Home</a>", unsafe_allow_html=True)

# --- Home Page ---
if current_page == "Home":
    st.title("Agricultural ML Dashboard")
    st.write("Select a functionality:")
    st.markdown("""
    <div class="card-container">
      <a href="?page=Pest Detection" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Pest Detection">
          <div class="card-title">Pest Detection</div>
        </div>
      </a>
      <a href="?page=Disease Detection" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Disease Detection">
          <div class="card-title">Disease Detection</div>
        </div>
      </a>
      <a href="?page=Crop Recommendation" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Crop Recommendation">
          <div class="card-title">Crop Recommendation</div>
        </div>
      </a>
      <a href="?page=Fertilizer Recommendation" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1592982537447-7440770cbfc9?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Fertilizer Recommendation">
          <div class="card-title">Fertilizer Recommendation</div>
        </div>
      </a>
      <a href="?page=Irrigation Management" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1586771107445-d3ca888129ce?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Irrigation Management">
          <div class="card-title">Irrigation Management</div>
        </div>
      </a>
      <a href="?page=Weather App" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1516912481808-3406841bd33c?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Weather App">
          <div class="card-title">Weather App</div>
        </div>
      </a>
    </div>
    """, unsafe_allow_html=True)

# --- (Other pages would follow below; for brevity, only the Home page and logo are shown here) ---

# Example: If current_page is "Pest Detection", show that page:
elif current_page == "Pest Detection":
    st.title("Pest Detection Interface")
    render_back_button()
    # ... (rest of your Pest Detection code) ...

# (Other page sections go here as needed)