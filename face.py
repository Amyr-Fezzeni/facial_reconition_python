import numpy as np
import cv2
import pickle
import os
import pyttsx3
import serial

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)


def speak(audio):
    engine.startLoop(False)
    engine.say(audio)
    engine.iterate()
    engine.endLoop()


base_dir = os.path.dirname(os.path.abspath(__file__))
pickles_location = os.path.join(base_dir, 'pickles/face-labels.pickle')
training_location = os.path.join(base_dir, 'recognizers/face-trainner.yml')
cascade_location = os.path.join(
    base_dir, 'cascades/data/haarcascade_frontalface_alt2.xml')
print(pickles_location)
print(training_location)
face_cascade = cv2.CascadeClassifier(cascade_location)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(training_location)

labels = {"name", 1}
with open(pickles_location, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
    print(labels)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ser = serial.Serial(port="COM5", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
p = 250
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in face:
        print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = frame[y: y + h, x: x + w]

        id_, conf = recognizer.predict(roi_gray)
        print(str(int(conf)) + "%   ------     " + labels[id_])
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255, 255, 255)
        stroke = 2

        if 64 <= conf <= 85:

            cv2.putText(frame, name, (x + 30, y + h + 15), font,
                        1, color, stroke, cv2.LINE_AA)
        else:
            cv2.putText(frame, "unknown", (x + 30, y + h + 15), font,
                        1, color, stroke, cv2.LINE_AA)
        img_item = "simple.png"
        cv2.imwrite(img_item, roi_gray)
        # if x >270 :
        #     ser.write(b"4")
        # elif x <230 :
        #     ser.write(b"3")
        # if y >140 :
        #      ser.write(b"2")
        # elif y <100 :
        #      ser.write(b"1")
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

