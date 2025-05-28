
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Define class labels
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the trained model
model = load_model(r"C:\Users\HP\OneDrive\Desktop\Final weight\flower_classifier_model.keras")

# App title
st.title("🌸 Flower Classification")
st.write("Upload an image of a flower, and the model will predict its category.")

# Upload image
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((240, 240))  # Resize to match model input size
    img_array = np.array(img)
    img_array = img_array.reshape(1, 240, 240, 3) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = categories[predicted_class_index]
    confidence = np.max(predictions) * 100

    # Layout with image and result side by side
    col1, col2 = st.columns([1, 2])  # 1: Image, 2: Result

    with col1:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Display prediction result
        st.success(f"🌼 Predicted Flower: **{predicted_class.capitalize()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
