import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from PIL import Image


import os
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'

# Rest of your imports and code...
import streamlit as st
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
      grid-template-columns: repeat(3, 420px);
      gap: 40px;
      justify-content: center;
      margin-top: 20px;
      width: 100%;
    }
    .card {
      background-color: rgba(255, 255, 255, 0.25);
      border-radius: 10px;
      width: 420px;
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
    /* Back button styling with white background at 75% opacity */
    .back-button {
      display: inline-block;
      background-color: rgba(255, 255, 255, 0.75);
      color: #000000;
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

# Helper to render a back button (as a styled link)
def render_back_button():
    st.markdown("<a href='?page=Home' target='_self' class='back-button'>Back to Home</a>", unsafe_allow_html=True)

# --- Initialize session state for model loading status ---
if 'model_loading_status' not in st.session_state:
    st.session_state.model_loading_status = {
        'pest_model': False,
        'disease_model': False,
        'crop_model': False,
        'fertilizer_model': False
    }
    
# --- Home Page ---
if current_page == "Home":
    st.title("Agricultural ML Dashboard")
    st.write("Select a functionality:")

    # Home page cards arranged in a 3x2 grid
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

# --- Pest Detection Page ---
elif current_page == "Pest Detection":
    st.title("Pest Detection Interface")
    render_back_button()
    
    # Check if model file exists before attempting to load
    model_path = "pest_model.h5"
    if not os.path.exists(model_path):
        st.warning("⚠ Model file not found. Using demo mode instead.")
        st.session_state.model_loading_status['pest_model'] = False
    else:
        # Only load the model if it hasn't been loaded yet
        if not st.session_state.model_loading_status['pest_model']:
            try:
                import tensorflow as tf
                # Use proper session-based caching for the model
                @st.cache_resource
                def load_pest_model():
                    return tf.keras.models.load_model(model_path)
                
                with st.spinner("Loading pest detection model..."):
                    pest_model = load_pest_model()
                st.session_state.model_loading_status['pest_model'] = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loading_status['pest_model'] = False
        
    class_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 
                   'mites', 'mosquito', 'sawfly', 'stem_borer']

    st.write("Upload an image of a pest to detect:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pest")
    
    if uploaded_file is not None:
        # Efficiently process the image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image for prediction
        if st.session_state.model_loading_status['pest_model']:
            import tensorflow as tf
            try:
                # More efficient image processing
                image_resized = image.resize((225, 225))
                image_array = np.array(image_resized) / 255.0  # Normalize image
                image_array = np.expand_dims(image_array, axis=0)
                
                with st.spinner("Analyzing image..."):
                    pest_model = load_pest_model()
                    predictions = pest_model.predict(image_array)
                
                predicted_index = np.argmax(predictions)
                predicted_class = class_names[predicted_index]
                confidence = np.max(predictions)
                
                st.success("Analysis complete!")
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Demo mode - random prediction
            import random
            predicted_class = random.choice(class_names)
            confidence = random.uniform(0.7, 0.95)
            
            st.info("🔍 DEMO MODE: Model not available, showing sample result")
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

# --- Disease Detection Page ---
elif current_page == "Disease Detection":
    st.title("Plant Disease Detection")
    render_back_button()
    
    # Check if model file exists
    model_path = "model.h5"
    if not os.path.exists(model_path):
        st.warning("⚠ Model file not found. Using demo mode instead.")
        st.session_state.model_loading_status['disease_model'] = False
    else:
        # Only load the model if it hasn't been loaded yet
        if not st.session_state.model_loading_status['disease_model']:
            try:
                import tensorflow as tf
                # Use proper session-based caching for the model
                @st.cache_resource
                def load_disease_model():
                    return tf.keras.models.load_model(model_path)
                
                with st.spinner("Loading disease detection model..."):
                    disease_model = load_disease_model()
                st.session_state.model_loading_status['disease_model'] = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loading_status['disease_model'] = False
    
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
        # Display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image for prediction
        if st.session_state.model_loading_status['disease_model']:
            import tensorflow as tf
            try:
                # Resize and process image
                image_resized = image.resize((128, 128))
                input_arr = np.array(image_resized) / 255.0  # Normalize
                input_arr = np.expand_dims(input_arr, axis=0)
                
                with st.spinner("Analyzing leaf image..."):
                    disease_model = load_disease_model()
                    predictions = disease_model.predict(input_arr)
                
                predicted_index = np.argmax(predictions, axis=1)[0]
                predicted_class = class_labels[predicted_index]
                
                st.success("Analysis complete!")
                st.write("### Prediction:")
                st.write(f"{predicted_class}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Demo mode - random prediction
            import random
            predicted_class = random.choice(class_labels)
            
            st.info("🔍 DEMO MODE: Model not available, showing sample result")
            st.write("### Prediction:")
            st.write(f"{predicted_class}")

# --- Crop Recommendation Page ---
elif current_page == "Crop Recommendation":
    st.title("Crop Prediction System")
    render_back_button()
    st.write("Enter the required parameters to predict the best crop.")
    
    # Check if model file exists
    model_path = "mdl_crv1.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠ Model file not found. Using demo mode instead.")
        st.session_state.model_loading_status['crop_model'] = False
    else:
        # Only load the model if it hasn't been loaded yet
        if not st.session_state.model_loading_status['crop_model']:
            try:
                import joblib
                
                # Use proper session-based caching for the model
                @st.cache_resource
                def load_crop_model():
                    model = joblib.load(model_path)
                    if hasattr(model, 'estimators_'):
                        for estimator in model.estimators_:
                            if not hasattr(estimator, 'monotonic_cst'):
                                setattr(estimator, 'monotonic_cst', None)
                    return model
                
                with st.spinner("Loading crop recommendation model..."):
                    model_crop = load_crop_model()
                st.session_state.model_loading_status['crop_model'] = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loading_status['crop_model'] = False
    
    crop_dict = {
        0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
        5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
        10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
        15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
        20: 'jute', 21: 'coffee'
    }
    
    # Create input form
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, step=0.1)
        P = st.number_input("Phosphorus (P)", min_value=0.0, step=0.1)
        K = st.number_input("Potassium (K)", min_value=0.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", step=0.1)
    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, step=0.1)
    
    if st.button("Predict Crop"):
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                 columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        if st.session_state.model_loading_status['crop_model']:
            try:
                import joblib
                with st.spinner("Analyzing soil and climate data..."):
                    model_crop = load_crop_model()
                    prediction = model_crop.predict(input_data)[0]
                
                predicted_crop = crop_dict.get(prediction, "Unknown Crop")
                st.success(f"The recommended crop is: {predicted_crop}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Demo mode - random prediction
            import random
            predicted_crop = random.choice(list(crop_dict.values()))
            
            st.info("🔍 DEMO MODE: Model not available, showing sample result")
            st.success(f"The recommended crop is: {predicted_crop}")

# --- Fertilizer Recommendation Page ---
elif current_page == "Fertilizer Recommendation":
    st.title("Fertilizer Recommendation")
    render_back_button()
    st.write("Enter the following details to get a fertilizer recommendation:")
    
    # Check if model file exists
    model_path = "mdl_fr_v5.pkl"
    if not os.path.exists(model_path):
        st.warning("⚠ Model file not found. Using demo mode instead.")
        st.session_state.model_loading_status['fertilizer_model'] = False
    else:
        # Only load the model if it hasn't been loaded yet
        if not st.session_state.model_loading_status['fertilizer_model']:
            try:
                import joblib
                
                # Use proper session-based caching for the model
                @st.cache_resource
                def load_fertilizer_model():
                    return joblib.load(model_path)
                
                with st.spinner("Loading fertilizer recommendation model..."):
                    model_fert = load_fertilizer_model()
                st.session_state.model_loading_status['fertilizer_model'] = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.model_loading_status['fertilizer_model'] = False
    
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
    
    # Create input form with two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Temperature", min_value=0.0, value=26.0)
        humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=82.0)
        moisture = st.number_input("Moisture", min_value=0.0, value=25.0)
        nitrogen = st.number_input("Nitrogen", min_value=0.0, value=86.0)
    
    with col2:
        potassium = st.number_input("Potassium", min_value=0.0, value=41.0)
        phosphorous = st.number_input("Phosphorous", min_value=0.0, value=36.0)
        soil_type = st.selectbox("Soil Type", options=list(soil_dict.keys()))
        crop_type = st.selectbox("Crop Type", options=list(crop_dict.keys()))
    
    soil_type_val = soil_dict[soil_type]
    crop_type_val = crop_dict[crop_type]
    
    feature_names = np.array(['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type',
                              'Nitrogen', 'Potassium', 'Phosphorous'])
    
    input_data = pd.DataFrame([[temperature, humidity, moisture, soil_type_val, crop_type_val,
                                nitrogen, potassium, phosphorous]], columns=feature_names)
    
    if st.button("Predict Fertilizer"):
        if st.session_state.model_loading_status['fertilizer_model']:
            try:
                import joblib
                with st.spinner("Analyzing soil and crop data..."):
                    model_fert = load_fertilizer_model()
                    prediction = model_fert.predict(input_data)
                
                predicted_fertilizer = inv_fertilizer_dict.get(prediction[0], "Unknown")
                st.success(f"Recommended Fertilizer: {predicted_fertilizer}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            # Demo mode - random prediction
            import random
            predicted_fertilizer = random.choice(list(fertilizer_dict.keys()))
            
            st.info("🔍 DEMO MODE: Model not available, showing sample result")
            st.success(f"Recommended Fertilizer: {predicted_fertilizer}")

# --- Irrigation Management Page ---
elif current_page == "Irrigation Management":
    st.title("Irrigation Management")
    render_back_button()
    
    st.write("This module provides irrigation management recommendations based on current weather conditions.")
    
    # Create a simple demo interface
    st.subheader("Current Field Data")
    
    col1, col2 = st.columns(2)
    with col1:
        soil_moisture = st.slider("Current Soil Moisture (%)", 0, 100, 25)
        temperature = st.slider("Temperature (°C)", 0, 50, 28)
    with col2:
        humidity = st.slider("Humidity (%)", 0, 100, 65)
        last_rain = st.number_input("Days Since Last Rain", min_value=0, value=3)
    
    crop_type = st.selectbox("Crop Type", 
                           ["Rice", "Wheat", "Corn", "Cotton", "Sugarcane", "Vegetables"])
    
    if st.button("Get Irrigation Recommendation"):
        with st.spinner("Analyzing irrigation needs..."):
            # Simple rule-based system
            moisture_threshold = 30
            
            if soil_moisture < moisture_threshold:
                status = "Critical! Immediate irrigation required."
                color = "red"
            elif soil_moisture < moisture_threshold + 15:
                status = "Low! Schedule irrigation within 24 hours."
                color = "orange" 
            elif soil_moisture < moisture_threshold + 30:
                status = "Moderate. Monitor conditions."
                color = "blue"
            else:
                status = "Good. No irrigation needed at this time."
                color = "green"
            
            # Display recommendation
            st.markdown(f"<h3 style='color:{color};'>Status: {status}</h3>", unsafe_allow_html=True)
            
            # Additional recommendations based on crop type
            st.subheader("Recommended Actions:")
            if soil_moisture < moisture_threshold:
                st.write(f"- Apply irrigation to raise soil moisture to at least 60% for {crop_type}")
                st.write("- Consider applying irrigation during early morning or late evening to minimize evaporation")
            elif soil_moisture < moisture_threshold + 15:
                st.write(f"- Schedule irrigation within the next 24 hours for {crop_type}")
                st.write("- Monitor weather forecast for potential rainfall")
            else:
                st.write("- Continue monitoring soil moisture levels")
                st.write(f"- Next evaluation recommended in {max(1, int(soil_moisture/10)-2)} days")

# --- Weather App Page ---
elif current_page == "Weather App":
    st.title("Weather App")
    render_back_button()
    
    st.subheader("Enter Location")
    location = st.text_input("City/Region", "")
    
    if st.button("Get Weather") or location:
        with st.spinner("Fetching weather data..."):
            # Demo weather data
            import random
            import datetime
            
            today = datetime.datetime.now()
            
            # Mock weather data
            weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Thunderstorm", "Clear"]
            temperatures = [random.randint(20, 35) for _ in range(7)]
            humidity_values = [random.randint(40, 90) for _ in range(7)]
            wind_speeds = [random.randint(5, 25) for _ in range(7)]
            
            # Current weather
            current_temp = temperatures[0]
            current_condition = weather_conditions[random.randint(0, len(weather_conditions)-1)]
            current_humidity = humidity_values[0]
            current_wind = wind_speeds[0]
            
            # Display current weather
            st.subheader(f"Current Weather in {location if location else 'Your Area'}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature", f"{current_temp}°C")
            with col2:
                st.metric("Humidity", f"{current_humidity}%")
            with col3:
                st.metric("Wind", f"{current_wind} km/h")
                
            st.info(f"Condition: {current_condition}")
            
            # Agricultural impact
            st.subheader("Agricultural Impact")
            
            if current_temp > 30:
                st.warning("⚠ High temperature may increase water requirements for crops")
            elif current_temp < 15:
                st.warning("⚠ Low temperature may affect crop growth. Consider protective measures")
                
            if current_condition in ["Light Rain", "Thunderstorm"]:
                st.info("☔ Current rainfall may reduce irrigation needs")
            elif current_condition == "Sunny" and current_temp > 25:
                st.warning("⚠ High evaporation rate. Consider irrigation")
                
            # 7-day forecast
            st.subheader("7-Day Forecast")
            
            forecast_data = []
            for i in range(7):
                day = (today + datetime.timedelta(days=i)).strftime("%a, %b %d")
                condition = weather_conditions[random.randint(0, len(weather_conditions)-1)]
                temp = temperatures[i]
                forecast_data.append({"Day": day, "Condition": condition, "Temp": temp})
            
            # Create forecast table
            forecast_df = pd.DataFrame(forecast_data)
            st.table(forecast_df)