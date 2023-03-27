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

def get_predictions_load(X_test):
    predictions = model_load.predict(X_test)    
    predictions = np.argmax(predictions, axis=1)
    return predictions

model_load = load_model('model')
labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title('Fashion MNIST Image Recognizer')
st.write('classes')
st.write(labels, horizontal = True)
genre = st.radio(
    "choose one of the two",
('Draw by hand', 'Upload image'))

if genre == 'Draw by hand':
    st.markdown('''
    Try to write a image!
    ''')
    SIZE = 192
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas')

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
#         st.image(rescaled)
else:
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        
if st.button('Predict'):
    try:
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prediction = model_load.predict(fe_data(test_x).reshape(1, 28, 28))    
        predictions = np.argmax(prediction, axis=1)
        output_text = predictions[0]
        font_size = "36px"
        st.markdown("<h3 style='text-align: left; color: black; font-size: {};'>{}</h3>".format(font_size, labels[output_text]), unsafe_allow_html=True)
    except:
        pass
    try:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(img_array )
#         img_array = cv2.resize(img_array.astype('uint8'), (28, 28))
#         img_array.reshape(1, 28, 28)
#         predict = model_load.predict(img_array.reshape(1, 28, 28))    
#         predicts = np.argmax(predict, axis=1)
#         output_text = predicts[0]
#         font_size = "36px"
#         st.markdown("<h3 style='text-align: left; color: black; font-size: {};'>{}</h3>".format(font_size, output_text), unsafe_allow_html=True)
    except:
        pass
