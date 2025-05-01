
from flask import Flask, render_template, request, jsonify
from keras.models import load_model

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_applications.mobilenet_v2 import preprocess_input
from keras_applications.mobilenet_v2 import decode_predictions
from keras_applications.mobilenet_v2 import MobileNetV2

import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)

# load model for prediction
modelxception_aug = load_model("A_XCEPTION_AUG_64.h5")


UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("cnn.html")

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classification.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = "temp_image.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert image to RGB and save it
        img = Image.open(file_path).convert('RGB')
        now = datetime.now()
        predict_image_path = os.path.join(app.config['UPLOAD_FOLDER'], now.strftime("%d%m%y-%H%M%S") + ".png")
        img.save(predict_image_path, format="png")
        img.close()

        # Prepare image for prediction
        img = image.load_img(predict_image_path, target_size=(299, 299, 3))
        x = image.img_to_array(img)
        x = x / 127.5 - 1  # Normalize
        x = np.expand_dims(x, axis=0)

        # Predict using the model
        prediction_array = modelxception_aug.predict(x)

        # Extract prediction and confidence
        class_names = ['Daun Sehat', 'Layu Bakteri', 'Powdery Mildew', 'Virus Kuning']
        predicted_class = class_names[np.argmax(prediction_array)]
        confidence = '{:2.0f}%'.format(100 * np.max(prediction_array))

        # Render the template with results
        return render_template("classification.html",
                               img_path=predict_image_path,
                               predicted_class=predicted_class,
                               confidence=confidence)

    else:
        resp = jsonify({'message': 'Invalid file type'})
        resp.status_code = 400
        return resp

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)