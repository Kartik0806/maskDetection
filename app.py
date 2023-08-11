from flask import Flask, render_template, request
import cv2
from tensorflow import keras
from keras.models import load_model
from keras import preprocessing
from PIL import Image, ImageOps
import numpy as np
import tensorflowjs as tfjs
import os 

img_shape=(50,50)
model=keras.models.load_model('model.tf', compile=False)
model.compile(loss='binary_crossentropy',metrics=['accuracy'], optimizer='adam')

print("model is loaded")
app = Flask(__name__)
@app.route("/", methods=["GET","POST"])
def home():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == "POST":
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        image=Image.open(file)
        image=image.resize(img_shape)
        image = keras.utils.img_to_array(image)
        image=np.expand_dims(image,axis=0)
        predictions = model.predict(image)
        score = float(predictions[0])
        output=''
        if(score>0.5):
            output = 'The person is not wearing a mask'
        else:
            output = 'The person is wearing a mask'
        return render_template('sec.html',pred_output=output, user_image=file_path)
if __name__ == "__main__":
    app.run(threaded=False)