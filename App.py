import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Load the "Brain" you just trained
model = tf.keras.models.load_model("eye_disease_model.h5")
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# 2. Page Styling
st.set_page_config(page_title="Eye Disease Detector", layout="centered")
st.title("👁️ AI Eye Disease Detection System")
st.write("Upload a retinal scan image to get an instant diagnosis.")

# 3. Image Upload
uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 4. Pre-process the image for the model
    st.write("🔄 Analyzing...")
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # 5. Make Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 6. Show Results
    st.success(f"Diagnosis: **{result}**")

    st.info(f"Confidence Level: **{confidence:.2f}%**")
