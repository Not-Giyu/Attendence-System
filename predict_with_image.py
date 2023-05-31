from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import os
import time
import datetime
import numpy as np
def preprocess_input(x):
    x = x.astype('uint8')
    x = Image.fromarray(x)
    x = x.resize((256, 256))
    x = np.array(x)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    return x
model = load_model('model.h5')

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")
known_person = ["Elon Musk","Rema","Jeff Bezos","Modi","Tata"]
known_person = np.array(known_person)
known_person = known_person[np.newaxis, :]
attendance = []
# Check if the prediction matches the known person


def predict_person(img_path):
    # Load the image from the camera
    camera_image = cv2.imread(img_path)

    camera_image = preprocess_input(camera_image)
    # Make a prediction using the model
    prediction = model.predict(camera_image)
    predicted_index = np.argmax(prediction)
    
    predicted_person = known_person[0, predicted_index]
    attendance.append(predicted_person)
    print(attendance)

def watch_folder(folder_path):
    images = set()
    while True:
        new_images = set(os.listdir(folder_path)) - images
        for image in new_images:
            image_path = os.path.join(folder_path, image)
            predict_person(img_path=image_path)
        if new_images:
            images.update(new_images)
        time.sleep(1)

if os.path.isdir(fr"C:\Users\brsre\OneDrive\Desktop\Coding\py\Attendence-System\{date}"):
    pass
    folder = fr"C:\Users\brsre\OneDrive\Desktop\Coding\py\Attendence-System\{date}"
else:
        os.makedirs(fr"C:\Users\brsre\OneDrive\Desktop\Coding\py\Attendence-System\{date}")
        folder = fr"C:\Users\brsre\OneDrive\Desktop\Coding\py\Attendence-System\{date}"

        print(folder)

watch_folder(folder_path=folder)
predict_person("Gokul_0.jpg")