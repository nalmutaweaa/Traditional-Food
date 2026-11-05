import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Bahraini Traditional Food Detector")

model = YOLO("best.pt") #model

img_file = st.camera_input("Take a photo of a note") #open cam
st.file_uploader("Or upload a photo", type=["jpg", "png"]) #upload

if img_file is not None:
    img = Image.open(img_file)
    results = model(np.array(img), conf=0.25)  
    annotated = results[0].plot()                
    st.image(annotated[:, :, ::-1], caption="Prediction")  

    for box in results[0].boxes: #displlay
        name = model.names[int(box.cls)]
        st.write(name, round(float(box.conf), 2))
