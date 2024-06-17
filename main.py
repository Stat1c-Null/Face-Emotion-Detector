import cv2
from deepface import DeepFace

#Trained model to detect faces
face_detect_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Get webcam access
camVideo = cv2.VideoCapture(0)

squareColor = (180, 40, 160)

while True:
  _, img = camVideo.read()
  grayFilter = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Put gray filter over video
  #Detect faces 
  faces = face_detect_model.detectMultiScale(grayFilter, 1.1, 5)
  #For every face found, put square around it
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), squareColor, 2)
  cv2.imshow('img', img)

  #Press ESCAPE to close everything
  k = cv2.waitKey(30) & 0xff
  if k==27:
    break
  camVideo.release

