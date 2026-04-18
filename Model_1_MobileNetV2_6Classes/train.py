#====================================
# 1. Drive mounting & Data extraction
#====================================

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

zip_path_drive = '/content/drive/MyDrive/Thesis_Project/UWCD_Dataset.zip'
local_zip = '/content/data.zip'
local_extract = '/content/arlo_local_data'

if os.path.exists(zip_path_drive):
    print("Moving data to Colabbbb")
    shutil.copy(zip_path_drive, local_zip)

    if os.path.exists(local_extract): shutil.rmtree(local_extract)
    os.makedirs(local_extract)

    print("Extracting dataaa")
    !unzip -q {local_zip} -d {local_extract}

    def find_target_root(base_path):
        for root, dirs, files in os.walk(base_path):
            target_classes = ['battery', 'textiles', 'organic_waste']
            if any(cls in dirs for cls in target_classes):
                return root
        return None

    DATA_PATH = find_target_root(local_extract)

    if DATA_PATH:
        print(f"\nData is mounting at: {DATA_PATH}")
        print(f"Found these trashes: {os.listdir(DATA_PATH)}")
        for folder in sorted(os.listdir(DATA_PATH)):
            folder_path = os.path.join(DATA_PATH, folder)
            if os.path.isdir(folder_path):
                print(f"  - {folder}: {len(os.listdir(folder_path))} images")
    else:
        print("Found none hjc hjc")
else:
    print("Found none on Drive, check it out pls")

#=====================================
# 2. Augmentation & Creating Train/Val
#=====================================

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    base_dir = DATA_PATH
except NameError:
    base_dir = '/content/arlo_local_data'

print(f"Inputing data from: {base_dir}")

#Augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,           
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,             
    horizontal_flip=True,
    vertical_flip=False,         
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2          
)

# Generator Batch Size = 32
BS = 32

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42 # seed 42
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=BS,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"\n8 labels: {train_generator.class_indices}")
print(f"Total batch Train: {len(train_generator)}")
print(f"Total batch Val: {len(val_generator)}")

if train_generator.num_classes != 8:
    print(f"unmatch, check data again")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras import mixed_precision

#==================================================================
# 3. Model Architecture: MobileNetV2 + Custom Head + Regularization
#==================================================================

# 1. Mixed Precision (Training faster)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 2. Base Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# 3.
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# Global Average & Max Pooling 
gap = layers.GlobalAveragePooling2D()(x)
gmp = layers.GlobalMaxPooling2D()(x)
merged = layers.Concatenate()([gap, gmp])

x = layers.BatchNormalization()(merged)

# Dense layer 256
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(8, activation='softmax', dtype='float32')(x)

model = models.Model(inputs, outputs)

# 4. Warm-up with LR 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()
print("\nReady to train")

#========================================================
# 4. Training Execution: with Checkpoint & EarlyStopping
#========================================================

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1. Warm-up (5 Epoch)
print("Warm-up 5 Epoch...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    verbose=1
)

# Fine-tuning(30 Epoch)
print("\nUnfreeze 30 layers...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Learning Rate 5e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# 3. Callbacks 
drive_save_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'

callbacks = [
    ModelCheckpoint(
        drive_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint(
        'Stage_Final_Backup.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,             
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,              
        mode='max',
        min_lr=1e-7,
        verbose=1
    )
]

# 4. Training
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print(f"\nDone")
print(f"Model was saved at: {drive_save_path}")


