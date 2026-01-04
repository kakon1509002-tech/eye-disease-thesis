import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Clear Streamlit's internal memory cache
st.cache_resource.clear()

# 2. Load the model using a relative path
# Make sure the file on GitHub is named exactly 'eye_disease_model.h5'
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("eye_disease_model.h5")

model = load_my_model()

# 3. Define the classes (Must be in the same order as your training folders)
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# 4. User Interface
st.title("👁️ Eye Disease Detection System")
st.write("Upload a retinal fundus image to get a diagnosis and confidence score.")

uploaded_file = st.file_file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 5. Pre-processing (Must match your training script)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0  # Normalization

    # 6. Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Convert to probabilities

    # 7. Output Results
    result = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.subheader(f"Diagnosis: {result}")
    st.write(f"**Confidence Level: {confidence:.2f}%**")
    
    # Optional: Show a progress bar for the confidence
    st.progress(int(confidence))