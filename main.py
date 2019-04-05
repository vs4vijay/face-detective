#!/usr/bin/env python3

import numpy as np
import cv2


config = {
  'name': 'Face Detective'
}

# haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}/haarcascade_frontalface_default.xml')
eye_haar_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("[-] ERROR: Cannot open camera")
  exit()

def open_camera(cap):
  while(True):
    ret, frame = cap.read()

    processed_frame = process_frame(frame)

    # cv2.namedWindow(config['name'], cv2.WINDOW_NORMAL)
    processed_frame = cv2.resize(processed_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow(config['name'], processed_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
      break
    elif key == ord('s'):
      cv2.imwrite('temp/image-saved.png', processed_frame)

def process_frame(image_frame):

  image_frame = cv2.flip(image_frame, 1)

  detect_faces(image_frame, face_haar_cascade, eye_haar_cascade)

  return image_frame


def detect_faces(image_frame, classifier, next_classifier = None):

  gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

  detected_objs = classifier.detectMultiScale(gray_frame, scaleFactor = 1.3, minNeighbors = 5)
  for (x, y, w, h) in detected_objs:
    color = (0, 255, 0)
    cv2.rectangle(image_frame, (x, y), (x+w, y+h), color, 5)

    if next_classifier:
      child_frame = image_frame[y:y+h, x:x+w]
      child_gray_frame = gray_frame[y:y+h, x:x+w]

      nested_objs = next_classifier.detectMultiScale(child_gray_frame)
      for (_x, _y, _w, _h) in nested_objs:
        color = (0, 0, 255)
        cv2.rectangle(child_frame, (_x, _y), (_x+_w, _y+_h), color, 2)
  return image_frame


open_camera(cap)

cap.release()
cv2.destroyAllWindows()