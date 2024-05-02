from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app=Flask(__name__, template_folder='template')

# Load the trained model
model = load_model('vgg.h5')

# Function to preprocess image data
def preprocess_image(image_data):
    # Convert image data to PIL Image object
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image (e.g., resize, convert to array, normalize)
    # Replace this with your specific preprocessing steps based on your model requirements
    image = image.resize((32, 32))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values (assuming pixel range [0, 255])

    return image

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict')
def predict():
    # Hardcode an image for prediction (replace this with actual image data later)
    image_path = 'static/imgt1.jpg'
    image = Image.open(image_path)
    image_data = np.array(image)

    # Preprocess the image
    preprocessed_image = preprocess_image(image_data)

    # Perform prediction using the model
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))

    # Process prediction results (e.g., extract class label)
    # Replace this with your specific post-processing logic based on your model output
    predicted_class = np.argmax(prediction)

    # Return prediction result as JSON
    return render_template('test.html', data=predicted_class)


if __name__ == '__main__':
    app.run(debug=True)
