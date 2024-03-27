import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

from Emotion_Prediction import EMO
from Emotion_Prediction import image_EMO


st.title('Emotion detection')
if st.button('Predict your live emotion'):
        EMO()
else:
        Print("Error!")