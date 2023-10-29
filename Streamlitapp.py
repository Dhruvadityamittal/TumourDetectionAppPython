import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from predict import predict

if (__name__ == "__main__"):

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Tumour Detection System</h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    model_name = st.selectbox(
        'Select the Machine Learning Model',
        ["ResNet","GoogleNet","MobileNet"])

    uploaded_image = st.file_uploader("Upload the image of brain scan")

    if (st.button("Predict", type="primary")):
        
        if uploaded_image is not None:
            prediction = predict(model_name,uploaded_image)
            im = Image.open(uploaded_image)
            
            st.image(im)
            prediction = predict(model_name,uploaded_image)

            # st.write("Prediction Results " + prediction)
            st.write(f'<p style="color: blue; font-size: 200%">Prediction Results => {prediction}</p>', unsafe_allow_html=True)

            
        
        else:
            st.error('Please upload a valid image for tumour detection', icon="🚨")

        