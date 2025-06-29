import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# 1. Prepare Data Generators
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/train',  # Folder: data/train/biodegradable, recyclable, trash
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_test_gen.flow_from_directory(
    'data/val',  # Folder: data/val/biodegradable, recyclable, trash
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = val_test_gen.flow_from_directory(
    'data/test',  # Folder: data/test/biodegradable, recyclable, trash
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 2. Build the Transfer Learning Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train the Model
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# 4. Save the Trained Model
os.makedirs("model", exist_ok=True)
model.save("model/vgg16.h5")  # Save as vgg16.h5 so Flask can use it

# 5. Save Class Labels for Reference
labels = list(train_data.class_indices.keys())
with open("model/labels.txt", "w") as f:
    f.write("\n".join(labels))
