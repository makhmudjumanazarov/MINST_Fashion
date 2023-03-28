import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

def fe_data(df):
    df = df / 255.
    return df

model_load = load_model('model')
labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title('Fashion MNIST Image Recognizer')
st.write('classes')
st.write(labels, horizontal = True)

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
        
if st.button('Predict'):
    try:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(img_array)
        img_array = cv2.resize(img_array.astype('uint8'), (28, 28))
        predict = model_load.predict(fe_data(img_array).reshape(1, 28, 28))    
        predicts = np.argmax(predict, axis=1)
        output_text = predicts[0]
        font_size = "24px"
        st.markdown("<h3 style='text-align: left; color: black; font-size: {};'>{}</h3>".format(font_size, labels[output_text]), unsafe_allow_html=True)
    except:
        pass
