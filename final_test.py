# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import io
# from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input


# app = Flask(__name__, template_folder='template')

# # Load the trained model
# model = load_model('models/vgg.h5')
# inception_model_path = 'models/inceptionv3_brain_tumor.h5'
# inception_model = load_model(inception_model_path)

# def preprocess_image_inception_v3(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((75, 75))
#     image = np.array(image) / 255.0
#     return image


# # Function to preprocess image data
# def preprocess_image(image_data):
#     # Convert image data to PIL Image object
#     image = Image.open(io.BytesIO(image_data))

#     # Preprocess the image (e.g., resize, convert to array, normalize)
#     # Replace this with your specific preprocessing steps based on your model requirements
#     image = image.resize((32, 32))  # Resize image to match model input size
#     image = np.array(image) / 255.0  # Normalize pixel values (assuming pixel range [0, 255])

#     return image

# @app.route('/', methods=['GET'])
# def hello_world():
#     return render_template('home.html')


# @app.route('/', methods=['POST'])
# def predict():
#     mriImage = request.files['mriImage']
#     image_path = 'static/uploads/' + mriImage.filename
#     mriImage.save(image_path)

#     # Read the uploaded image
#     with open(image_path, 'rb') as f:
#         image_data = f.read()

#     # Preprocess the image for VGG model
#     preprocessed_image_vgg = preprocess_image(image_data)

#     # Perform prediction using the VGG model
#     prediction_vgg = model.predict(np.expand_dims(preprocessed_image_vgg, axis=0))

#     # Preprocess the image for InceptionV3 model
#     preprocessed_image_inception_v3 = preprocess_image_inception_v3(image_data)

#     # Perform prediction using the InceptionV3 model
#     prediction_inception_v3 = inception_model.predict(np.expand_dims(preprocessed_image_inception_v3, axis=0))

#     # Process prediction results for VGG model
#     predicted_class_vgg = np.argmax(prediction_vgg)
#     if predicted_class_vgg == 2:
#         predicted_class_vgg = 'No Tumour'
#     elif predicted_class_vgg == 1:
#         predicted_class_vgg = 'Meningioma Tumor'

#     # Process prediction results for InceptionV3 model
#     predicted_class_inception_v3 = np.argmax(prediction_inception_v3)
#     # Replace class labels for InceptionV3 model accordingly

#     return render_template('result.html', filename=image_path, predicted_class_vgg=predicted_class_vgg, predicted_class_inception_v3=predicted_class_inception_v3)


# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__, template_folder='template')

# # Load the trained models
# vgg_model = load_model('models/vgg.h5')
# inception_model = load_model('models/inceptionv3_brain_tumor.h5')
# cnn_model = load_model('models/cnn_brain_tumor.h5')

# # Preprocess image for VGG model
# def preprocess_image_vgg(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((32, 32))
#     image = np.array(image) / 255.0
#     return image

# # Preprocess image for InceptionV3 model
# def preprocess_image_inception_v3(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((75, 75))
#     image = np.array(image) / 255.0
#     return image

# # Preprocess image for CNN model
# def preprocess_image_cnn(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((32, 32))
#     image = np.array(image) / 255.0
#     return image

# @app.route('/', methods=['GET'])
# def hello_world():
#     return render_template('home.html')

# @app.route('/', methods=['POST'])
# def predict():
#     mriImage = request.files['mriImage']
#     image_path = 'static/uploads/' + mriImage.filename
#     mriImage.save(image_path)

#     # Read the uploaded image
#     with open(image_path, 'rb') as f:
#         image_data = f.read()

#     # Preprocess the image for VGG model
#     preprocessed_image_vgg = preprocess_image_vgg(image_data)
#     prediction_vgg = vgg_model.predict(np.expand_dims(preprocessed_image_vgg, axis=0))
#     predicted_class_vgg = np.argmax(prediction_vgg)

#     # Preprocess the image for InceptionV3 model
#     preprocessed_image_inception_v3 = preprocess_image_inception_v3(image_data)
#     prediction_inception_v3 = inception_model.predict(np.expand_dims(preprocessed_image_inception_v3, axis=0))
#     predicted_class_inception_v3 = np.argmax(prediction_inception_v3)

#     # Preprocess the image for CNN model
#     preprocessed_image_cnn = preprocess_image_cnn(image_data)
#     prediction_cnn = cnn_model.predict(np.expand_dims(preprocessed_image_cnn, axis=0))
#     predicted_class_cnn = np.argmax(prediction_cnn)

#     return render_template('result.html', filename=image_path, predicted_class_vgg=predicted_class_vgg, 
#                            predicted_class_inception_v3=predicted_class_inception_v3, predicted_class_cnn=predicted_class_cnn)

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, session, url_for, redirect
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import io
# import os
# import cv2
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Flatten, Dropout, Dense
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import UserMixin
# import requests



# app = Flask(__name__, template_folder='template')



# # Load the trained models
# vgg_model = load_model('models/vgg.h5')
# inception_model = load_model('models/inceptionv3_brain_tumor.h5')
# cnn_model = load_model('models/cnn_brain_tumor.h5')
# resnet_model = load_model('models/resnet_brain_tumor.h5')

# # Preprocess image for VGG model
# def preprocess_image_vgg(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((32, 32))
#     image = np.array(image) / 255.0
#     return image

# # Preprocess image for InceptionV3 model
# def preprocess_image_inception_v3(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((75, 75))
#     image = np.array(image) / 255.0
#     return image

# # Preprocess image for CNN model
# def preprocess_image_cnn(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((32, 32))
#     image = np.array(image) / 255.0
#     return image

# # Preprocess image for ResNet model
# def preprocess_image_resnet(image_data):
#     image = Image.open(io.BytesIO(image_data))
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     return image

# @app.route('/', methods=['GET'])
# def display_first_page():
#     return render_template('home.html')

# Route to handle signup form submission
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         role = 'user'

#         signup_data = {
#             "username": username,
#             "email": email,
#             "password": password,
#             "role": role
#         }

#         signup_url = 'https://fitnessapp-666y.onrender.com/api/signup'
#         response = requests.post(signup_url, json=signup_data)

#         if response.status_code == 200:
#             session['username'] = username  # Store username in session
#             return redirect(url_for('hello_world'))
#         else:
#             error_message = 'Signup failed. Please try again.'
#             return render_template('signup.html', error=error_message)

#     return render_template('signup.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']

#         login_data = {
#             "email": email,
#             "password": password
#         }

#         login_url = 'https://fitnessapp-666y.onrender.com/api/login'
#         response = requests.post(login_url, json=login_data)

#         if response.status_code == 200:
#             session['email'] = email  # Store email in session
#             return redirect(url_for('home'))
#         else:
#             error_message = 'Login failed. Please check your credentials.'
#             return render_template('login.html', error=error_message)

#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.pop('username', None)  # Remove username from session
#     session.pop('email', None)  # Remove email from session
#     return redirect(url_for('hello_world'))


# @app.route('/home', methods=['POST'])
# def predict():
#     mriImage = request.files['mriImage']
#     image_path = 'static/uploads/' + mriImage.filename
#     mriImage.save(image_path)

#     # Read the uploaded image
#     with open(image_path, 'rb') as f:
#         image_data = f.read()

#     # Preprocess the image for VGG model
#     preprocessed_image_vgg = preprocess_image_vgg(image_data)
#     prediction_vgg = vgg_model.predict(np.expand_dims(preprocessed_image_vgg, axis=0))
#     predicted_class_vgg = np.argmax(prediction_vgg)

#     # Preprocess the image for InceptionV3 model
#     preprocessed_image_inception_v3 = preprocess_image_inception_v3(image_data)
#     prediction_inception_v3 = inception_model.predict(np.expand_dims(preprocessed_image_inception_v3, axis=0))
#     predicted_class_inception_v3 = np.argmax(prediction_inception_v3)

#     # Preprocess the image for CNN model
#     preprocessed_image_cnn = preprocess_image_cnn(image_data)
#     prediction_cnn = cnn_model.predict(np.expand_dims(preprocessed_image_cnn, axis=0))
#     predicted_class_cnn = np.argmax(prediction_cnn)
    
#     # Preprocess the image for ResNet model
#     preprocessed_image_resnet = preprocess_image_resnet(image_data)
#     prediction_resnet = resnet_model.predict(np.expand_dims(preprocessed_image_resnet, axis=0))
#     predicted_class_resnet = np.argmax(prediction_resnet)

#     return render_template('result.html', filename=image_path, predicted_class_vgg=predicted_class_vgg, 
#                            predicted_class_inception_v3=predicted_class_inception_v3, predicted_class_cnn=predicted_class_cnn,
#                            predicted_class_resnet=predicted_class_resnet)

# if __name__ == '__main__':
#     app.run(debug=True)
