import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = os.path.join("model_output", "best_model.h5")

img_height, img_width = 224, 224
CLASS_NAMES = [
    'Bird_drop', 'Clean', 'Dusty', 'Eletrical_damage',
    'Physical_Damage', 'Snow_covered'
]

try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(image_path):
    if model is None:
        return "Model not loaded", 0.0

    try:
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction[0]) * 100

        return predicted_class_name, f"{confidence:.2f}"
    except Exception as e:
        return f"Error processing image: {e}", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            return render_template('result.html', prediction=prediction, confidence=confidence, filename=filename)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
        app.run(debug=True)
