import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from googletrans import Translator  # unofficial Google Translate API

# --- Helper: Translation Function ---
def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

# --- Sidebar: Language Options for Page Translation ---
language_options = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Malayalam': 'ml'
}
st.sidebar.header("Page Translation")
translate_page = st.sidebar.checkbox("Translate page content?")
target_language = st.sidebar.selectbox("Select target language", list(language_options.keys()), index=0)

# --- CSS Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
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
      margin-top: 20px;
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

# --- Navigation via Query Parameters ---
query_params = st.experimental_get_query_params()
if "page" in query_params:
    current_page = query_params["page"][0]
else:
    current_page = "Home"

# Helper to render a back button
def render_back_button():
    st.markdown("<a href='?page=Home' target='_self' class='back-button'>Back to Home</a>", unsafe_allow_html=True)

# --- Static Page Texts (for Home page translation) ---
page_title = "Agricultural ML Dashboard"
page_description = "Select a functionality:"

if translate_page and language_options[target_language] != 'en':
    dest_lang = language_options[target_language]
    page_title = translate_text(page_title, dest_lang)
    page_description = translate_text(page_description, dest_lang)

# --- Home Page ---
if current_page == "Home":
    st.title(page_title)
    st.write(page_description)
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
          <img src="https://cdn.britannica.com/89/126689-004-D622CD2F/Potato-leaf-blight.jpg" alt="Disease Detection">
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
      <a href="?page=Weather App" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://images.unsplash.com/photo-1516912481808-3406841bd33c?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" alt="Weather App">
          <div class="card-title">Weather App</div>
        </div>
      </a>
      <a href="?page=Translate" target="_self" style="text-decoration: none; color: inherit;">
        <div class="card">
          <img src="https://play-lh.googleusercontent.com/ZrNeuKthBirZN7rrXPN1JmUbaG8ICy3kZSHt-WgSnREsJzo2txzCzjIoChlevMIQEA=w240-h480-rw" alt="Translate">
          <div class="card-title">Translate</div>
        </div>
      </a>
    </div>
    """, unsafe_allow_html=True)

# --- Pest Detection Page ---
elif current_page == "Pest Detection":
    st.title("Pest Detection Interface")
    render_back_button()
    
    @st.cache(allow_output_mutation=True)
    def load_pest_model():
        return tf.keras.models.load_model("pest_model.h5")
    pest_model = load_pest_model()
    class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 
                   'mites', 'mosquito', 'sawfly', 'stem_borer']
    st.write("Upload an image of a pest to detect:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pest")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_resized = image.resize((225, 225))
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        predictions = pest_model.predict(image_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions)
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

# --- Disease Detection Page ---
elif current_page == "Disease Detection":
    st.title("Plant Disease Detection")
    render_back_button()
    @st.cache(allow_output_mutation=True)
    def load_disease_model():
        return tf.keras.models.load_model("model.h5")
    disease_model = load_disease_model()
    class_labels = [
        'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
        'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
        'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
        'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
        'Grape_Esca(Black_Measles)', 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy',
        'Orange_Haunglongbing(Citrus_greening)', 'Peach_Bacterial_spot', 'Peach__healthy',
        'Pepper,bell_Bacterial_spot', 'Pepper,bellhealthy', 'Potato__Early_blight',
        'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy',
        'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite',
        'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
    ]
    st.write("Upload an image of a plant leaf to detect the disease:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="disease")
    if uploaded_file is not None:
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(128, 128))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = disease_model.predict(input_arr)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_index]
        st.write("### Prediction:")
        st.write(f"{predicted_class}")

# --- Crop Recommendation Page ---
elif current_page == "Crop Recommendation":
    st.title("Crop Prediction System")
    render_back_button()
    st.write("Enter the required parameters to predict the best crop.")
    model_crop = joblib.load("mdl_crv1.pkl")
    if hasattr(model_crop, 'estimators_'):
        for estimator in model_crop.estimators_:
            if not hasattr(estimator, 'monotonic_cst'):
                setattr(estimator, 'monotonic_cst', None)
    crop_dict = {
        0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
        5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
        10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
        15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
        20: 'jute', 21: 'coffee'
    }
    N = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1)
    P = st.number_input("Phosphorus (P)", min_value=0.0, step=0.1)
    K = st.number_input("Potassium (K)", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    if st.button("Predict Crop"):
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        prediction = model_crop.predict(input_data)[0]
        predicted_crop = crop_dict.get(prediction, "Unknown Crop")
        st.success(f"The recommended crop is: {predicted_crop}")

# --- Fertilizer Recommendation Page ---
elif current_page == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    render_back_button()
    st.write("Enter the following details to get a fertilizer recommendation:")
    model_fert = joblib.load('mdl_fr_v5.pkl')
    soil_dict = {'Clayey': 0, 'Loamy': 1, 'Red': 2, 'Black': 3, 'Sandy': 4}
    crop_dict = {
        'rice': 0, 'Wheat': 1, 'Tobacco': 2, 'Sugarcane': 3, 'Pulses': 4,
        'pomegranate': 5, 'Paddy': 6, 'Oil seeds': 7, 'Millets': 8, 'Maize': 9,
        'Ground Nuts': 10, 'Cotton': 11, 'coffee': 12, 'watermelon': 13,
        'Barley': 14, 'kidneybeans': 15, 'orange': 16
    }
    fertilizer_dict = {
        'Urea': 0, 'TSP': 1, 'Superphosphate': 2, 'Potassium sulfate.': 3,
        'Potassium chloride': 4, 'DAP': 5, '28-28': 6, '20-20': 7,
        '17-17-17': 8, '15-15-15': 9, '14-35-14': 10, '14-14-14': 11,
        '10-26-26': 12, '10-10-10': 13
    }
    inv_fertilizer_dict = {v: k for k, v in fertilizer_dict.items()}
    feature_names = np.array(['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type',
                               'Nitrogen', 'Potassium', 'Phosphorous'])
    temperature = st.number_input("Temparature", min_value=0.0, value=26.0)
    humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=82.0)
    moisture = st.number_input("Moisture", min_value=0.0, value=25.0)
    nitrogen = st.number_input("Nitrogen", min_value=0.0, value=86.0)
    potassium = st.number_input("Potassium", min_value=0.0, value=41.0)
    phosphorous = st.number_input("Phosphorous", min_value=0.0, value=36.0)
    soil_type = st.selectbox("Soil Type", options=list(soil_dict.keys()))
    crop_type = st.selectbox("Crop Type", options=list(crop_dict.keys()))
    soil_type_val = soil_dict[soil_type]
    crop_type_val = crop_dict[crop_type]
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type_val, crop_type_val,
                                nitrogen, potassium, phosphorous]], columns=feature_names)
    if st.button("Predict Fertilizer"):
        prediction = model_fert.predict(input_data)
        predicted_fertilizer = inv_fertilizer_dict.get(prediction[0], "Unknown")
        st.success(f"Recommended Fertilizer: {predicted_fertilizer}")

# --- Weather App Page ---
elif current_page == "Weather App":
    st.title("Weather App")
    render_back_button()
    st.write("Display current weather data and predictions.")
    st.write("Current Weather: Sunny, 25°C")
    st.write("Forecast: Clear skies with mild winds.")

# --- Translate Page ---
elif current_page == "Translate":
    st.title("Google Translate")
    render_back_button()
    st.write("Enter text to translate and select a target language:")
    translator = Translator()
    input_text = st.text_area("Text to Translate", height=150)
    target_lang = st.selectbox("Target Language", options=list(language_options.keys()))
    if st.button("Translate"):
        if input_text:
            dest_lang = language_options[target_lang]
            translation = translator.translate(input_text, dest=dest_lang)
            st.markdown("**Translated Text:**")
            st.write(translation.text)
        else:
            st.error("Please enter text to translate.")
