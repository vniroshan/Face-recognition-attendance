import cv2
import numpy as np
import face_recognition

imgNiro = face_recognition.load_image_file("imageBasic/niro.jpg")
imgNiro = cv2.cvtColor(imgNiro,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("imageBasic/niro1.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgNiro)[0]
encodeNiro = face_recognition.face_encodings(imgNiro)[0]
cv2.rectangle(imgNiro,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)



faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeNiroTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeNiro],encodeNiroTest)
faceDis = face_recognition.face_distance([encodeNiro],encodeNiroTest)
print(results,faceDis)

cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow('Niro', imgNiro)
cv2.imshow('Niro Test', imgTest)
cv2.waitKey(0)