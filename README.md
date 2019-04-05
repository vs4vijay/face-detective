# FaceDetective
A Face Detection &amp; Recognization Program based on Python &amp; OpenCV


## Pre-requisites
- Python 3
- OpenCV `brew install opencv`

## Installation
`python3 -m pip install -r requirements.txt`


## Running
`python3 main.py`


## Usage
- Pressing key `s` will save the image in `temp/` folder
- Pressing key `q` or `ESC` will close the program





#######################################

### Rough Work Below

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')



for path, subdirnames, filenames in os.walk("trainingImages")


cv2.face.LBPHFaceRecognizer_create()