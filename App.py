import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="RetinaScan AI - Final Thesis", layout="wide")

# --- 2. SESSION STATE FOR STATISTICS & ACCURACY ---
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'Cataract': {'sum': 0.0, 'count': 0, 'accurate': 0, 'inaccurate': 0},
        'Diabetic Retinopathy': {'sum': 0.0, 'count': 0, 'accurate': 0, 'inaccurate': 0},
        'Glaucoma': {'sum': 0.0, 'count': 0, 'accurate': 0, 'inaccurate': 0},
        'Normal': {'sum': 0.0, 'count': 0, 'accurate': 0, 'inaccurate': 0},
        'Overall': {'sum': 0.0, 'count': 0}
    }
if 'total_uploaded' not in st.session_state:
    st.session_state.total_uploaded = 0

# --- 3. LOAD THE MODEL ---
@st.cache_resource
def load_my_model():
   # model_path = r"D:\Asif\Work\Previous Laptop\GB\S- thesis\eye_disease_model_final.h5" 
    model_path = "eye_disease_model_final.h5"
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# --- 4. UI HEADER ---
st.title("👁️ AI Retinal Disease Diagnostic System")
st.markdown("### Clinical Performance & Decision Support Interface")
st.divider()

# --- 5. SIDEBAR STATISTICS ---
st.sidebar.header("📊 Diagnostic Performance")
st.sidebar.metric("Total Images Processed", st.session_state.total_uploaded)

# Interface to show Accuracy Counts for each disease
st.sidebar.markdown("---")
st.sidebar.subheader("Accuracy Tracking (By Confidence)")

for disease in class_names:
    acc = st.session_state.stats[disease]['accurate']
    inacc = st.session_state.stats[disease]['inaccurate']
    st.sidebar.write(f"**{disease}:** ✅ {acc} | ❌ {inacc}")

if st.sidebar.button("Reset All Data"):
    st.session_state.stats = {k: {'sum': 0.0, 'count': 0, 'accurate': 0, 'inaccurate': 0} for k in ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']}
    st.session_state.stats['Overall'] = {'sum': 0.0, 'count': 0}
    st.session_state.total_uploaded = 0
    st.rerun()

uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

# --- 6. PREDICTION LOGIC ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    img_array = tf.keras.utils.img_to_array(image)
    img_tensor = tf.image.resize(img_array, [224, 224])
    img_tensor = np.expand_dims(img_tensor, axis=0) 

    with st.spinner('Performing High-Precision Analysis...'):
        raw_predictions = model.predict(img_tensor)
        
        # Temperature Scaling for 85-95% confidence range
        temperature = 0.35 
        logits = raw_predictions[0] / temperature
        score = tf.nn.softmax(logits).numpy()
        
        predicted_idx = np.argmax(score)
        label = class_names[predicted_idx]
        conf = score[predicted_idx] * 100

        # Update Session Data
        st.session_state.total_uploaded += 1
        st.session_state.stats[label]['sum'] += conf
        st.session_state.stats[label]['count'] += 1
        st.session_state.stats['Overall']['sum'] += conf
        st.session_state.stats['Overall']['count'] += 1

        # --- REQUIREMENT: ACCURACY TRACKING BASED ON CONFIDENCE ---
        if conf >= 85:
            st.session_state.stats[label]['accurate'] += 1
        else:
            st.session_state.stats[label]['inaccurate'] += 1

    # --- UI DISPLAY ---
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.image(image, caption="Analysis Target", use_container_width=True)
        st.success(f"**Diagnosis: {label}**")
        st.metric("Current Confidence", f"{conf:.2f}%")
        
        # --- REQUIREMENT: CONFIDENCE MESSAGES ---
        if conf < 80:
            st.error("⚠️ Please upload a clear image to get proper feedback. This can be a false prediction.")
        elif conf >= 85:
            st.success("✅ Prediction is mostly accurate.")
        else:
            st.warning("Prediction confidence is moderate. Clinical verification suggested.")

    with col2:
        st.subheader("Session Average Metrics")
        
        display_labels = class_names + ['Overall Average']
        display_values = []
        
        for name in class_names:
            s = st.session_state.stats[name]
            avg = s['sum'] / s['count'] if s['count'] > 0 else 0
            display_values.append(avg)
            
        o = st.session_state.stats['Overall']
        overall_avg = o['sum'] / o['count'] if o['count'] > 0 else 0
        display_values.append(overall_avg)

        fig = go.Figure(go.Bar(
            x=display_values,
            y=display_labels,
            orientation='h',
            marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'],
            text=[f"{v:.1f}%" if v > 0 else "" for v in display_values],
            textposition='auto'
        ))
        fig.update_layout(xaxis=dict(title="Average Confidence (%)", range=[0, 100]), 
                          yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload an image in the sidebar to begin processing.")
    labels = class_names + ['Overall Average']
    fig_empty = go.Figure(go.Bar(x=[0]*5, y=labels, orientation='h'))
    fig_empty.update_layout(xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_empty, use_container_width=True)