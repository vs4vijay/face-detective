#!/usr/bin/env python3

import numpy as np
import cv2


config = {
  'name': 'Face Detective'
}


haar_cascade_face = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("[-] ERROR: Cannot open camera")
  exit()

def open_camera(cap):
  while(True):
    ret, frame = cap.read()

    processed_frame = process_frame(frame)

    cv2.namedWindow(config['name'], cv2.WINDOW_NORMAL)
    cv2.imshow('frame', processed_frame)

    # cv2.waitKey(1) == 27
    # cv.waitKey(0)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == 27):
          break

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