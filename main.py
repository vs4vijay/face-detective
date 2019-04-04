import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def open_camera(cap):
	while(True):
	    ret, frame = cap.read()

	    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    gray_frame = cv2.flip(gray_frame, 1)

	    cv2.imshow('frame', gray_frame)

	    # cv2.waitKey(1) == 27
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

open_camera(cap)

cap.release()
cv2.destroyAllWindows()