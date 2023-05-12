import cv2
import numpy as np
import os
# Load the training data
training_data = np.load('training_data.npy')

# Create a classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Train the classifier
classifier.train(training_data)

# Create a function to detect gestures
def detect_gesture(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = classifier.detectMultiScale(gray, 1.1, 4)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Get the largest face in the image
    face = faces[0]

    # Get the coordinates of the face
    x, y, w, h = face

    # Crop the image to the face
    face = image[y:y+h, x:x+w]

    # Resize the face to 200x200 pixels
    face = cv2.resize(face, (200, 200))

    # Preprocess the face
    face = cv2.equalizeHist(face)

    # Convert the face to a NumPy array
    face = np.array(face)

    # Predict the gesture
    prediction = classifier.predict(face)

    # Return the prediction
    return prediction

# Create a function to play a sound based on the gesture
def play_sound(gesture):
    # If the gesture is a thumbs up, play the "thumbs up" sound
    if gesture == 1:
        cv2.imshow('Thumbs Up', cv2.imread('thumbs_up.png'))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        os.system('play thumbs_up.mp3')

    # If the gesture is a peace sign, play the "peace sign" sound
    elif gesture == 2:
        cv2.imshow('Peace Sign', cv2.imread('peace_sign.png'))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        os.system('play peace_sign.mp3')

    # If the gesture is a rock on, play the "rock on" sound
    elif gesture == 3:
        cv2.imshow('Rock On', cv2.imread('rock_on.png'))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        os.system('play rock_on.mp3')

# Start the video capture
cap = cv2.VideoCapture(0)

# While the video is running
while True:
    # Get a frame from the video
    ret, frame = cap.read()

    # If the frame is not empty
    if ret == True:
        # Detect gestures in the frame
        gesture = detect_gesture(frame)

        # If a gesture is detected
        if gesture is not None:
            # Play a sound based on the gesture
            play_sound(gesture)

    # If the user presses the `q` key, stop the video capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all open windows
cv2.destroyAllWindows()
