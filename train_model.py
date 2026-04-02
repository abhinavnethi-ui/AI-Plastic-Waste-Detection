import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image size
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16

# Load images from folders
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_data_gen.flow_from_directory(
    'images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_data_gen.flow_from_directory(
    'images',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Build improved CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train with better parameters
model.fit(train_generator, validation_data=validation_generator, epochs=20, verbose=1)

# Save model
os.makedirs("train_model", exist_ok=True)
model.save("train_model/plastic_model.h5")

# Save labels in index order so predictions map correctly
class_indices = train_generator.class_indices  # {class_name: index}
labels = [None] * len(class_indices)
for class_name, class_index in class_indices.items():
	labels[class_index] = class_name

with open("train_model/labels.json", "w", encoding="utf-8") as f:
	json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
