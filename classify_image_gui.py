import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ===============================
# 1️⃣ Configuration
# ===============================
# Path to your trained model
MODEL_PATH = os.path.join("model_output", "best_model.h5")

# Define image dimensions
img_height, img_width = 224, 224

# Define the class names in the correct order (usually alphabetical)
# This order MUST match the one used during training.
# You can verify this by checking the output of `train_generator.class_indices` in your training script.
CLASS_NAMES = [
    'Bird_drop',
    'Clean',
    'Dusty',
    'Eletrical_damage',
    'Physical_Damage',
    'Snow_covered'
]

# ===============================
# 2️⃣ Load the Trained Model
# ===============================
try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure 'best_model.h5' is in the 'model_output' directory and you have run the training script first.")
    model = None

# ===============================
# 3️⃣ Image Prediction Function
# ===============================
def predict_image(image_path):
    """Loads an image, preprocesses it, and returns the model's prediction."""
    if model is None:
        return "Model not loaded.", 0.0

    # Load and resize the image
    img = load_img(image_path, target_size=(img_height, img_width))
    # Convert image to a numpy array
    img_array = img_to_array(img)
    # Rescale the image pixels (from 0-255 to 0-1)
    img_array /= 255.0
    # Expand dimensions to create a batch of 1
    img_batch = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_batch)

    # Get the class with the highest probability
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction[0]) * 100  # Convert to percentage

    return predicted_class_name, confidence

# ===============================
# 4️⃣ GUI Application
# ===============================
def select_and_predict_image():
    """Opens a file dialog to select an image and updates the GUI."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return # User cancelled the dialog

    # Display the selected image
    img = Image.open(file_path)
    img.thumbnail((350, 350)) # Resize for display
    photo = ImageTk.PhotoImage(img)
    lbl_image.config(image=photo)
    lbl_image.image = photo # Keep a reference

    # Get prediction
    prediction, confidence = predict_image(file_path)

    # Update the result label
    result_text = f"Prediction: {prediction}"
    confidence_text = f"Confidence: {confidence:.2f}%"
    lbl_result.config(text=result_text)
    lbl_confidence.config(text=confidence_text)


# --- Setup the main window ---
root = tk.Tk()
root.title("Solar Panel Fault Detection")
root.geometry("500x550")
root.configure(bg='#f0f0f0')

# --- Create and place widgets ---
# Main Frame
main_frame = Frame(root, bg='#f0f0f0')
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

# Image display label 🖼️
lbl_image = Label(main_frame, bg='white', relief='sunken', borderwidth=2)
lbl_image.pack(pady=10, fill="both", expand=True)

# Select Image Button
btn_select = Button(
    main_frame,
    text="Select Image to Classify",
    command=select_and_predict_image,
    font=('Helvetica', 12, 'bold'),
    bg='#4CAF50',
    fg='white'
)
btn_select.pack(pady=10, fill='x')

# Prediction Result Label
lbl_result = Label(
    main_frame,
    text="Prediction: N/A",
    font=('Helvetica', 14, 'bold'),
    bg='#f0f0f0'
)
lbl_result.pack(pady=5)

# Confidence Score Label
lbl_confidence = Label(
    main_frame,
    text="Confidence: N/A",
    font=('Helvetica', 12),
    bg='#f0f0f0'
)
lbl_confidence.pack(pady=5)

# --- Start the GUI event loop ---
root.mainloop()