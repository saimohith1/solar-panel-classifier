# solar_panel_fault_detection.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===============================
# 1️⃣ Paths & Parameters
# ===============================
dataset_dir = "Faulty_solar_panel"  # Corrected folder name to match your description
img_height, img_width = 224, 224
batch_size = 32
epochs = 20
learning_rate = 1e-4
# MODIFIED: Updated the number of classes from 3 to 6 to match your subfolders.
# The classes are: Bird_drop, Clean, Dusty, Eletrical_damage, Physical_Damage, Snow_covered
num_classes = 6

# ===============================
# 2️⃣ Data Preparation
# ===============================
# This assumes your 'faulty_solar_panel' directory contains 'train' and 'val' subdirectories,
# and each of those contains the 6 class folders.
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' # Added fill_mode for better augmentation
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Typically, you don't need to shuffle validation data
)

# ===============================
# 3️⃣ Load Pre-trained VGG16
# ===============================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
# This final Dense layer now correctly uses num_classes=6 for its output
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary() # Added model summary to see the architecture

# ===============================
# 4️⃣ Callbacks
# ===============================
# It's good practice to save the model in a specific folder
os.makedirs("model_output", exist_ok=True)
checkpoint_path = os.path.join("model_output", "best_model.h5")

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ===============================
# 5️⃣ Train Model
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stop],
    # Adding steps_per_epoch and validation_steps is good practice
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# ===============================
# 6️⃣ Plot Accuracy & Loss
# ===============================
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig(os.path.join("model_output", "training_plots.png")) # Save the plots
plt.show()