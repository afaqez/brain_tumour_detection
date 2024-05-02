# import numpy as np
# import cv2
# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing import image
# from keras.layers import Flatten

# # Load the trained ResNet50 model
# from keras.models import load_model
# model = load_model('models/cnn_brain_tumor.h5')  # Replace 'path_to_your_trained_model.h5' with the actual path

# # Function to preprocess the input image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# # Function to make predictions
# def predict_image(image_path):
#     # Preprocess the image
#     img = preprocess_image(image_path)
#     # Predict
#     prediction = model.predict(img)
#     # Decode the prediction
#     class_labels = ['notumor', 'glioma', 'pituitary', 'meningioma']
#     predicted_label = class_labels[np.argmax(prediction)]
#     return predicted_label

# # Path to the image you want to predict
# image_path = 'static/uploads/img.jpg'  # Replace 'path_to_your_image.jpg' with the actual path

# # Make prediction
# prediction = predict_image(image_path)
# print("Predicted label:", prediction)



# WORKING CODE FOR INCEPTIONV3


# import numpy as np
# import cv2
# from keras.preprocessing import image
# from keras.models import load_model
# from keras.applications.inception_v3 import preprocess_input
# import os


# model_path = 'models/inceptionv3_brain_tumor.h5'  
# model = load_model(model_path)

# # Function to preprocess the input image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(75, 75))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# # Function to make predictions
# def predict_image(image_path):
#     img = preprocess_image(image_path)
#     prediction = model.predict(img)
#     print(prediction)
#     class_labels = ['class1', 'class2', 'class3', 'class4']  # i have no idea of the classes 
#     predicted_label = class_labels[np.argmax(prediction)]
#     return predicted_label


# image_path = 'static/uploads/img.jpg' 
# prediction = predict_image(image_path)
# print("Predicted label:", prediction)



# ===================================================

# WORKING CODE OF CNN MODEL

# import numpy as np
# import cv2
# from keras.preprocessing import image
# from keras.models import load_model
# import os

# # Load the trained model
# model_path = 'models/cnn_brain_tumor.h5'
# model = load_model(model_path)

# # Function to preprocess the input image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(32, 32))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0  # Normalize pixel values
#     return img_array

# # Function to make predictions
# def predict_image(image_path):
#     img = preprocess_image(image_path)
#     prediction = model.predict(img)
#     print(prediction)
#     class_labels = ['class1', 'class2', 'class3', 'class4']  # Replace with your actual class labels
#     predicted_label = class_labels[np.argmax(prediction)]
#     return predicted_label

# # Path to the image you want to test
# image_path = 'static/uploads/image(2).jpg'  # Update with the path to your image
# prediction = predict_image(image_path)
# print("Predicted label:", prediction)


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the model from the .h5 file
model = load_model('models/resnet_brain_tumor.h5')

# Define a function for preprocessing the image data
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

# Path to the image you want to test
image_path = 'static/uploads/image(2).jpg'

# Preprocess the image
img = preprocess_image(image_path)

# Perform prediction
prediction = model.predict(img)
print(prediction)
# Convert the prediction to human-readable format (e.g., class labels)
# Here, you can use np.argmax() or any other method to get the predicted class
# For example, if you have 4 classes, you can use class_labels = ['Class1', 'Class2', 'Class3', 'Class4']
# predicted_class = class_labels[np.argmax(prediction)]
# Replace class_labels with your actual class labels
class_labels = ['class1', 'class2', 'class3', 'class4']
predicted_class = class_labels[np.argmax(prediction)]
# predicted_class = np.argmax(prediction)

print("Predicted Class:", predicted_class)