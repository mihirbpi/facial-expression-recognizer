import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp

# Data Loading
DATA_PATH = "data/"

# Data & Augmentation
NUM_CLASSES = 3
CLASS_NAMES = ["happy", "sad", "angry"]
NUM_LANDMARKS = 1404

# Model 
MODEL_VERSION = "v1"

# Saving/Loading Model + Testing
MODEL_PATH = "model/"
MODEL_SAVE_PATH = MODEL_PATH + MODEL_VERSION + "/"
MODEL_NAME = "model.pt"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

class NeuralNet(nn.Module):
    def __init__(self, num_classes, input_size):
        super(NeuralNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, self.num_classes)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)

    
    def forward(self, x):
        out = x
        out = F.relu(self.batch_norm1(self.fc1(out)))
        out = F.relu(self.batch_norm2(self.fc2(out)))
        out = F.relu(self.batch_norm3(self.fc3(out)))
        out = F.relu(self.batch_norm4(self.fc4(out)))
        out = F.relu(self.batch_norm5(self.fc5(out)))
        out = F.relu(self.fc6(out))
        out = F.softmax(out, dim=1)
        return out

model = NeuralNet(num_classes=NUM_CLASSES, input_size=NUM_LANDMARKS).to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH + MODEL_NAME))
model.eval()

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("loaded mp model")
# Initializing the drawing utils for drawing the landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(1)

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
      
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
      
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
      
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
    # Drawing Face Land Marks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS
    )
          
    # Display the resulting image
    cv2.imshow("Face Landmarks", image)

    # Listen for keypress
    key = cv2.waitKey(1) & 0xFF

    # If key 'p' is pressed predict the facial expression
    if key == ord('p'):
        if (results.face_landmarks):
            landmarks_list = []

            for l in results.face_landmarks.landmark:
                landmarks_list.append(l.x)
                landmarks_list.append(l.y)
                landmarks_list.append(l.z)
            landmarks_tensor = torch.from_numpy(np.array(landmarks_list)).to(torch.float32).reshape(-1,NUM_LANDMARKS)
            predicted_class_idx = torch.max(model(landmarks_tensor), 1).indices.item()
            print(CLASS_NAMES[predicted_class_idx],"\n")
            continue
    #  If key 'q' is pressed quit from the loop
    elif key == ord('q'):
        break

# When the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()


