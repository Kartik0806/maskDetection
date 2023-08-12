import streamlit as st
import cv2
import keras
from keras.models import load_model
from keras import preprocessing
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

st.header('Mask Detector')

def main():
    file_uploaded=st.file_uploader('Choose the file', type=['jpg','jpeg','jpg','png'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        image = image.convert('RGB')
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        st.pyplot(figure)
def predict_class(image):
    model=keras.models.load_model('customModel.h5',compile=False)
    # model=keras.models.load_model('modelnew.h5',compile=False)
    model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='adam')
    img_shape=(50,50)
    image=image.resize(img_shape)
    image = np.asarray(image)
    image=np.expand_dims(image,axis=0)
    predictions = model.predict(image)
    score = float(predictions[0])
    output=''
    if(score>0.5):
        output = 'The person is not wearing a mask'
    else:
        output = 'The person is wearing a mask'
    return output

if __name__=="__main__":
    main()
