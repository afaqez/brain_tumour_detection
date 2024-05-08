from flask import Flask, render_template, request, session, url_for, json, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dropout, Dense
import requests
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='template')
CORS(app)  

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  

# Load the trained models
vgg_model = load_model('models/vgg.h5')
inception_model = load_model('models/inceptionv3_brain_tumor.h5')
cnn_model = load_model('models/cnn_brain_tumor.h5')
resnet_model = load_model('models/resnet_brain_tumor.h5')

# Preprocess image for VGG model
def preprocess_image_vgg(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return image
    

# Preprocess image for InceptionV3 model
def preprocess_image_inception_v3(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((75, 75))
    image = np.array(image) / 255.0
    return image

# Preprocess image for CNN model
def preprocess_image_cnn(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return image

# Preprocess image for ResNet model
def preprocess_image_resnet(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

@app.route('/upload', methods=['POST'])
def predict():
    print("Received request at /upload endpoint")  

    if 'mriImage' not in request.files:
        print("No file found in the request")  
        return jsonify({'error': 'No file uploaded'})

    mriImage = request.files['mriImage']
    image_path = 'static/uploads/' + mriImage.filename
    mriImage.save(image_path)

    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_data = np.array(image_data)

    preprocessed_image_vgg = preprocess_image_vgg(image_data)
    prediction_vgg = vgg_model.predict(np.expand_dims(preprocessed_image_vgg, axis=0))
    predicted_class_vgg = np.argmax(prediction_vgg)

    preprocessed_image_inception_v3 = preprocess_image_inception_v3(image_data)
    prediction_inception_v3 = inception_model.predict(np.expand_dims(preprocessed_image_inception_v3, axis=0))
    predicted_class_inception_v3 = np.argmax(prediction_inception_v3)

    preprocessed_image_cnn = preprocess_image_cnn(image_data)
    prediction_cnn = cnn_model.predict(np.expand_dims(preprocessed_image_cnn, axis=0))
    predicted_class_cnn = np.argmax(prediction_cnn)

    preprocessed_image_resnet = preprocess_image_resnet(image_data)
    prediction_resnet = resnet_model.predict(np.expand_dims(preprocessed_image_resnet, axis=0))
    predicted_class_resnet = np.argmax(prediction_resnet)

    prediction_data = {
        'filename': image_path,
        'predicted_class_vgg': int(predicted_class_vgg),
        'predicted_class_inception_v3': int(predicted_class_inception_v3),
        'predicted_class_cnn': int(predicted_class_cnn),
        'predicted_class_resnet': int(predicted_class_resnet)
    }

    print("Prediction data:", prediction_data)  

    # Save prediction data to result.txt, clearing the file first
    with open('result.txt', 'w') as file:
        file.write(json.dumps(prediction_data))

    return jsonify(prediction_data) 

if __name__ == '__main__':
    app.run(port=5000, debug=True)
