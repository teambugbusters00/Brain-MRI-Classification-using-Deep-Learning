
import streamlit as st
import requests
from PIL import Image
import io
import subprocess
import time
import os

st.set_page_config(page_title='Brain Tumor Classifier', layout='centered')
st.title('🧠 Brain Tumor Classification')
st.write('FastAPI Backend + Streamlit Frontend')

# Start FastAPI in the background if it's not running (Local/Spaces helper)
if 'backend_started' not in st.session_state:
    subprocess.Popen(['uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8000'])
    st.session_state['backend_started'] = True
    time.sleep(2) # Give it a second to start

uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        # Convert to bytes for request
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", files={"file": byte_im})
            result = response.json()
            
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']*100:.2f}%")
            st.json(result['all_probabilities'])
        except Exception as e:
            st.error(f"Error connecting to FastAPI backend: {e}")
