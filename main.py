#!/usr/bin/env python3

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def open_camera(cap):
  while(True):
    ret, frame = cap.read()

    processed_frame = process_frame(frame)

    cv2.imshow('frame', processed_frame)

    # cv2.waitKey(1) == 27
    # cv.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

def process_frame(image_frame):
  gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

  gray_frame = cv2.flip(gray_frame, 1)

  final_frame = detect_faces(gray_frame)
  return final_frame


def detect_faces(image_frame):
  return image_frame


open_camera(cap)

cap.release()
cv2.destroyAllWindows()