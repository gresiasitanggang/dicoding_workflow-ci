"""modelling.ipynb

## Import Semua Packages/Library yang Digunakan
"""

#Import Library

import os
import zipfile
import numpy as np
import pandas as pd
import shutil
import random
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Input
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
import mlflow
import mlflow.tensorflow
mlflow.tensorflow.autolog()

"""## Data Preparation

### Data Loading
"""

zip_path = 'archive.zip'
extracted_dir = 'dataset'

if not os.path.exist(extracted_dir) and os.path.exists(zip_path):
    print("Mengekstrak dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Ekstrak selesai.")

original_dataset_dir = os.path.join(extracted_dir, "train", "train")
base_dir = "dataset_split"

"""### Data Preprocessing

#### Split Dataset
"""

#Data Split
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

classes = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]

for cls in classes:
    cls_path = os.path.join(original_dataset_dir, cls)
    images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.seed(42)
    random.shuffle(images)

    # Membagi dataset: 70% Train, 15% Validation, 15% Test
    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

    def copy_images(img_list, src_dir, dst_dir):
        for img in img_list:
            shutil.copy(
                os.path.join(src_dir, img),
                os.path.join(dst_dir, img)
            )

    copy_images(train_imgs, cls_path, os.path.join(train_dir, cls))
    copy_images(val_imgs, cls_path, os.path.join(val_dir, cls))
    copy_images(test_imgs, cls_path, os.path.join(test_dir, cls))

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 4

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
)

class_names = list(train_generator.class_indices.keys())
print(class_names)

print("Train dir:", train_generator.directory)
print("Val dir  :", val_generator.directory)
print("Test dir :", test_generator.directory)

"""## Modelling"""

#CNN Sequential
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

base_model.trainable = False

model = Sequential([
    base_model,
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(6, activation='softmax')
])

#Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#Callback
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    verbose=1
)

#Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

"""## Evaluasi dan Visualisasi"""

#Evaluasi Model

#Train
train_loss, train_acc = model.evaluate(train_generator, verbose=1)
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Train Loss     : {train_loss:.4f}")

#Validation
val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Validation Loss     : {val_loss:.4f}")

#Test
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")