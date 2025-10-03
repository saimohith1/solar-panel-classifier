import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ===============================
# 1. Application Initialization
# ===============================
app = Flask(__name__)

# --- Configuration ---
# Set a folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set the path to your trained model
MODEL_PATH = os.path.join("model_output", "best_model.h5")

# Define image dimensions and class names (MUST match training script)
img_height, img_width = 224, 224
CLASS_NAMES = [
    'Bird_drop', 'Clean', 'Dusty', 'Eletrical_damage',
    'Physical_Damage', 'Snow_covered'
]

# --- Load the Model ---
# Load the model once at startup for efficiency
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ===============================
# 2. Prediction Logic
# ===============================
def predict_image(image_path):
    """Loads, preprocesses an image and returns the model's prediction."""
    if model is None:
        return "Model not loaded", 0.0

    try:
        # Load and resize the image
        img = load_img(image_path, target_size=(img_height, img_width))
        # Convert image to a numpy array and rescale
        img_array = img_to_array(img) / 255.0
        # Expand dimensions to create a batch of 1
        img_batch = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_batch)

        # Get the class with the highest probability and its confidence
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction[0]) * 100

        return predicted_class_name, f"{confidence:.2f}"
    except Exception as e:
        return f"Error processing image: {e}", 0.0

# ===============================
# 3. Web Application Routes
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles both displaying the upload form and processing the uploaded image."""
    if request.method == 'POST':
        # Check if a file was posted
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            # Secure the filename to prevent malicious paths
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get the prediction
            prediction, confidence = predict_image(filepath)
            
            # Render the result page
            return render_template('result.html', prediction=prediction, confidence=confidence, filename=filename)

    # For a GET request, just show the upload form
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image file to be displayed on the result page."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ===============================
# 4. Run the Application
# ===============================
if __name__ == '__main__':
    # Setting debug=True is useful for development as it provides detailed error pages
    # and automatically reloads the server when you make changes.
    app.run(debug=True)