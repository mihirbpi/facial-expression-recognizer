import numpy as np
import cv2
import mediapipe as mp

DATA_PATH = "data/"
NUM_DATA_POINTS = 50 # number of data points to collect for each class
NUM_CLASSES = 3
CLASS_NAMES = ["happy", "sad", "angry"]

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

for i in range(NUM_CLASSES):
    
    for j in range(NUM_DATA_POINTS):
      print(f"{i+1},{j+1}")

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
          # Enter key 's' to save the image
          if cv2.waitKey(1) & 0xFF == ord('s'):
              
              if (results.face_landmarks):
                landmarks_list = []

                for l in results.face_landmarks.landmark:
                   landmarks_list.append(l.x)
                   landmarks_list.append(l.y)
                   landmarks_list.append(l.z)
                landmarks_array = np.array(landmarks_list)
                print(landmarks_array)
                print(landmarks_array.shape)
                np.save(f"{DATA_PATH}{i+1}_{j+1}.npy", landmarks_array)
                break
 
# When the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()


