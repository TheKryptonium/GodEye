import cv2
import face_recognition
import math
import numpy as np
import os, sys


def faceConfidence(faceDistance, faceThresh=0.6):
    range = 1 - faceThresh
    linearVal = (1 - faceDistance)/(range*2)
    
    if faceDistance > faceThresh:
        return str(round(linearVal*100, 2)) + "%"
    else:
        value = (linearVal + ((1-linearVal)*math.pow((linearVal-0.5)*2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class faceRecognition:
    faceLocations = []
    faceEncodings = []
    faceNames = []
    knownEncodings = []
    knownNames = []
    currentFrame = True

    def __init__(self):
        self.encodeFaces()

    def encodeFaces(self):
        for image in os.listdir('images'):
            faceImage = face_recognition.load_image_file('images/{}'.format(image))
            faceEncoding = face_recognition.face_encodings(faceImage)[0]

            self.knownEncodings.append(faceEncoding)
            self.knownNames.append((image.split('.'))[0])
        
        print(self.knownNames)
        

    def runRecognition(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)

        if not cap.isOpened():
            sys.exit('Impossible de lire les donnees de la camera')

        while True:
            ret, frame = cap.read()

            if self.currentFrame:
                smallFrame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgbFrame = smallFrame[:,:,::-1]

                self.faceLocations = face_recognition.face_locations(rgbFrame)
                self.faceEncodings = face_recognition.face_encodings(rgbFrame, self.faceLocations)

                self.faceNames = []
                for encoding in self.faceEncodings:
                    matches = face_recognition.compare_faces(self.knownEncodings,encoding)
                    name = 'Inconnu'
                    confidence = "Inconnu" 

                    distances = face_recognition.face_distance(self.knownEncodings, encoding)
                    bestMatch = np.argmin(distances) 

                    if matches[bestMatch]:
                        name = self.knownNames[bestMatch]
                        confidence = faceConfidence(distances[bestMatch])
                    
                    self.faceNames.append(f'{name} ({confidence})')

            self.currentFrame = not self.currentFrame

            for (top, left, bottom, right), name in zip(self.faceLocations, self.faceNames):
                top*=4
                left*=4
                bottom*=4
                right*=4
                
                if name == 'Inconnu':
                    color = (0,0,255)
                else:
                    color = (0,255,0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('God$Eye', frame)
            key = cv2.waitKey(1)

            if key == ord('q') or key==27:
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    fr = faceRecognition()
    fr.runRecognition()