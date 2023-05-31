import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Function to duplicate an image a given number of times
def duplicate_image(image_path, num_duplicates, save_path):
    image = Image.open(image_path)
    for i in range(num_duplicates):
        new_image = image.copy()
        new_image.save(os.path.join(save_path, f'Tata_{i}.jpg'))

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    label_index = 0
    for filename in os.listdir(folder):
        person_folder = os.path.join(folder, filename)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                img = Image.open(image_path)
                img = img.resize((256, 256))
                img_array = keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(label_index)
            label_index += 1
    return images, labels, label_index

# Load images and labels from dataset folder
images, labels, label_index = load_images_from_folder('dataset')

# Split data into training and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

# Convert data to tensors
train_images = tf.stack(train_images, axis=0)
train_labels = tf.convert_to_tensor(train_labels)

test_images = tf.stack(test_images, axis=0)
test_labels = tf.convert_to_tensor(test_labels)

# Create training and test datasets

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)

# Create an instance of the ImageDataGenerator class
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Fit the data generator on your training data
datagen.fit(train_images)

# Create base model using Xception pre-trained model
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(256, 256, 3),
    include_top=False)

base_model.trainable = True

# Define model architecture
inputs = keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=True)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Conv2D(64, 3, activation='relu',padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Conv2D(128, 3, activation='relu',padding='same', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(label_index)(x)
model = keras.Model(inputs, outputs)

# Compile model
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Create an instance of the ImageDataGenerator class with data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Fit the data generator on your training data
datagen.fit(train_images)
    
# Train model using early stopping
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(datagen.flow(train_images, train_labels),
                    validation_data=(test_images, test_labels),
                    epochs=20,
                    callbacks=[early_stopping])

# Evaluate model on test dataset
test_loss,test_acc=model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')

# Save model
model.save('model.h5')