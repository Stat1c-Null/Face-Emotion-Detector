import cv2
from deepface import DeepFace

#Trained model to detect face
face_detect_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Get webcam access
cam = cv2.VideoCapture(0)
squareColor = (180, 40, 160)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
  _, img = cam.read()
  grayFilter = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_detect_model.detectMultiScale(grayFilter, 1.3, 5)
  for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), squareColor, 2)
  cv2.imshow('img', img)

  #Detect emotions, age, gender, ethnicity
  try:
    result = DeepFace.analyze(img, actions=["emotion", "age", "gender", "race"])
    print("-----------")
    print(result[0]["dominant_emotion"])
    print(result[0]["age"])
    print(result[0]["dominant_gender"])
    print(result[0]["dominant_race"])
    print("------------")

    emotion = result[0]["dominant_emotion"]
    age = result[0]["age"]
    gender = result[0]["dominant_gender"]
    race = result[0]["dominant_race"]

    emotion, age, gender, race =  str(emotion), str(age), str(gender), str(race)

    cv2.putText(img, emotion, (0, 50), font, 2, (255, 255, 255), 3)
    cv2.putText(img, age, (0, 100), font, 2, (255, 255, 255), 3)
    cv2.putText(img, gender, (0, 150), font, 2, (255, 255, 255), 3)
    cv2.putText(img, race, (0, 200), font, 2, (255, 255, 255), 3)
  except Exception as e:
    print("Error happend: ", e)

  #Press escape to close
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
  cam.release
