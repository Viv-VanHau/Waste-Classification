# ===================================
# 1. RANDOM SHUFFLE & TRAIN/VAL SPLIT
# ===================================
!pip install split-folders 
import os
import zipfile
import splitfolders
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator

drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/Thesis_Project/UWCD_10Classes_Final.zip'
extract_path = '/content/UWCD_10Classes_Raw'
final_dataset_path = '/content/UWCD_Split'  

if not os.path.exists(extract_path):
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Done")

data_dir = extract_path
for root, dirs, files in os.walk(extract_path):
    if len(dirs) == 10:
        data_dir = root
        break

if not os.path.exists(final_dataset_path):
    print("🔀 Đang xáo trộn ngẫu nhiên (Shuffle) và chia tỷ lệ 80 Train / 20 Val...")
    # seed=42 
    splitfolders.ratio(data_dir, output=final_dataset_path,
                       seed=42, ratio=(0.8, 0.2), group_prefix=None)
    print("Done splitting")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("\nInputting Train")
train_generator = train_datagen.flow_from_directory(
    os.path.join(final_dataset_path, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

print("\nInputting Val")
val_generator = val_datagen.flow_from_directory(
    os.path.join(final_dataset_path, 'val'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =================================
# 2. DATA IMBALANCE - CLASS WEIGHTS
# =================================
import numpy as np
from sklearn.utils import class_weight

print("Class Weights setting...")

train_classes = train_generator.classes

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)

class_weight_dict = dict(enumerate(class_weights))

print("\nDone buffing")
for class_idx, weight in class_weight_dict.items():
    class_name = list(train_generator.class_indices.keys())[class_idx]
    if "Grade_A" in class_name:
        print(f"Class {class_name:<20} : Weight = {weight:.4f} (buffed)")
    else:
        print(f"Class {class_name:<20} : Weight = {weight:.4f}")

# ==========================================
# 3. TRANING
# ==========================================
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("Inputing MobileNetV2from Model 1...")

model_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
base_model = load_model(model_path)

x = base_model.layers[-2].output

predictions = Dense(10, activation='softmax', name='Dense_10_Classes')(x)

model_stage2 = Model(inputs=base_model.input, outputs=predictions)

for layer in model_stage2.layers[:-1]: 
    layer.trainable = False

model_stage2.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

checkpoint = ModelCheckpoint('/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("Start FINE-TUNING...")
history = model_stage2.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop]
)

print("Done")


