import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from datetime import datetime

# Title
st.set_page_config(page_title="Heart Sound Analyzer", layout="centered")
st.title("🫀 Heart Sound Analyzer")

# New Case Section
st.sidebar.header("📝 New Case Entry")
with st.sidebar.form(key='patient_form'):
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    submit_button = st.form_submit_button(label='Create New Case')

if submit_button:
    st.success(f"New case created for {patient_name}, Age: {age}, Gender: {gender}")

# File Upload
uploaded_file = st.file_uploader("Upload a Heart Sound File (.wav)", type="wav")

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    # Display Audio Player
    st.audio(uploaded_file, format='audio/wav')

    # Display waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Heart Sound Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Dummy Prediction Placeholder
    st.subheader("🧠 AI Prediction")
    st.write("This is a placeholder for heart sound classification result.")
    st.success("Result: Normal")

    # Generate PDF report (simplified text report for now)
    report_text = f"""
    🫀 Heart Sound Analysis Report

    Patient Name: {patient_name}
    Age: {age}
    Gender: {gender}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ✅ Result: Normal
    🔍 Analysis: No abnormal murmur detected. Further clinical correlation advised.
    """

    # Convert to bytes and download link
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="Heart_Sound_Report.txt">📄 Download Report as Text</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Upload a heart sound (.wav) file to begin analysis.")
    
