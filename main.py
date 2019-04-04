#!/usr/bin/env python3

import numpy as np
import cv2


config = {
  'name': 'Face Detective'
}


# haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
haar_cascade_face = cv2.CascadeClassifier(f'{cv2.data.haarcascades}/haarcascade_frontalface_default.xml')


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
    cv2.imshow('frame', processed_frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
      break
    elif key == ord('s'):
      cv2.imwrite('temp/image-saved.png', processed_frame)

def process_frame(image_frame):
  gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

  final_frame = cv2.flip(gray_frame, 1)

  final_frame = detect_objects(final_frame, haar_cascade_face)
  return final_frame


def detect_objects(image_frame, classifier):
  detected_objs = classifier.detectMultiScale(image_frame, scaleFactor = 1.3, minNeighbors = 5)
  for (x, y, w, h) in detected_objs:
    color = (0, 255, 0)
    cv2.rectangle(image_frame, (x, y), (x+w, y+h), color, 2)
  return image_frame


open_camera(cap)

cap.release()
cv2.destroyAllWindows()