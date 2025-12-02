import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

MODEL_PATH = os.path.join("model_output", "best_model.h5")

img_height, img_width = 224, 224

CLASS_NAMES = [
    'Bird_drop',
    'Clean',
    'Dusty',
    'Eletrical_damage',
    'Physical_Damage',
    'Snow_covered'
]

try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_model.h5' is in the 'model_output' directory")
    model = None

def predict_image(image_path):
    if model is None:
        return "Model not loaded.", 0.0

    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction[0]) * 100 
    return predicted_class_name, confidence

def select_and_predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return 
        
    img = Image.open(file_path)
    img.thumbnail((350, 350)) 
    photo = ImageTk.PhotoImage(img)
    lbl_image.config(image=photo)
    lbl_image.image = photo 

    prediction, confidence = predict_image(file_path)

    result_text = f"Prediction: {prediction}"
    confidence_text = f"Confidence: {confidence:.2f}%"
    lbl_result.config(text=result_text)
    lbl_confidence.config(text=confidence_text)

root = tk.Tk()
root.title("Solar Panel Fault Detection")
root.geometry("500x550")
root.configure(bg='#f0f0f0')

main_frame = Frame(root, bg='#f0f0f0')
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

lbl_image = Label(main_frame, bg='white', relief='sunken', borderwidth=2)
lbl_image.pack(pady=10, fill="both", expand=True)

btn_select = Button(
    main_frame,
    text="Select Image to Classify",
    command=select_and_predict_image,
    font=('Helvetica', 12, 'bold'),
    bg='#4CAF50',
    fg='white'
)
btn_select.pack(pady=10, fill='x')

lbl_result = Label(
    main_frame,
    text="Prediction: N/A",
    font=('Helvetica', 14, 'bold'),
    bg='#f0f0f0'
)
lbl_result.pack(pady=5)

lbl_confidence = Label(
    main_frame,
    text="Confidence: N/A",
    font=('Helvetica', 12),
    bg='#f0f0f0'
)
lbl_confidence.pack(pady=5)

root.mainloop()
