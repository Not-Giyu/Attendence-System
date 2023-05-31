import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pyttsx3
from tensorflow.keras.preprocessing import image
import random
import os

engine = pyttsx3.init()

def shout(text):
    engine.say(text)
    engine.runAndWait()

# Load the saved model
model = load_model('model.h5')

# Create a dictionary that maps class labels to names
label_to_name = {0:"Elon Musk",
                 1:"Rema",
                 2:"Jeff Bezos",
                 3:"Modi",
                 4:"Tata"}

# Create a list of people who are present
present = []

# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        img = Image.fromarray(face)
        img = img.resize((224, 224))
        img = np.array(img) / 255.0

        # Make a prediction using the trained model
        prediction = model.predict(np.expand_dims(img, axis=0))

        # Get the predicted class
        predicted_class = np.argmax(prediction[0])


        # Get the name corresponding to the predicted class
        name = label_to_name[predicted_class]

        # Draw a rectangle around the face and display the name
        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (R, G, B), 2)
        font = cv2.FONT_ITALIC
        cv2.putText(frame, name, (x, y-10), font, 1, (R, G, B), 2)
        if name in present:
            pass
        else:
            present.append(name)
            print(present)
            shout(f'{name} is present')



    # Show the frame
    cv2.imshow('frame', frame)

    # Check if the user pressed the 'q' key

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
