import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Bahraini Traditional Food Detector!🍚")

model = YOLO("best.pt")

img_file = st.file_uploader("Or upload a photo", type=["jpg", "png"])

if img_file is not None:
    img = Image.open(img_file)
    results = model(np.array(img), conf=0.25)  
    annotated = results[0].plot()                
    st.image(annotated, caption="Prediction")    
    
    for box in results[0].boxes:
        name = model.names[int(box.cls)]
        
