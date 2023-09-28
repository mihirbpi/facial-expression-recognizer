import numpy as np
import cv2
import mediapipe as mp

DATA_PATH = "data/"

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

for i in range(3):
    for j in range(20):
      print(f"{i},{j}")
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
      
          # Drawing Right hand Land Marks
          mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
          )
          
          # Display the resulting image
          # cv2.namedWindow('Right Hand Landmarks', cv2.WINDOW_AUTOSIZE)
          cv2.imshow("Right Hand Landmarks", image)
          # Enter key 'q' to break the loop
          if cv2.waitKey(1) & 0xFF == ord('q'):
              if (results.right_hand_landmarks):
                cv2.imwrite(f"{DATA_PATH}{i}_{j}.png", image)
                landmarks_list = []
                for l in results.right_hand_landmarks.landmark:
                   landmarks_list.append(l.x)
                   landmarks_list.append(l.y)
                   landmarks_list.append(l.z)
                landmarks_array = np.array(landmarks_list)
                print(landmarks_array)
                print(landmarks_array.shape)
                break
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()


