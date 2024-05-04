import cv2
# from mtcnn import MTCNN
from matplotlib import pyplot, axes
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import numpy as np
import pickle
import os

if not os.path.exists('face_data'):
    os.makedirs('face_data')


def detect_faces(input_source):
    detector = MTCNN()
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', 640, 480)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    if input_source == 'webcam':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            num_faces = len(faces)
            cv2.putText(frame, f'Number of faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if faces != []:
                for person in faces:
                    traced_rectangle = person['box']
                    key_points = person['keypoints']
                '''cv2.rectangle(frame, (traced_rectangle[0], traced_rectangle[1]), (
                    traced_rectangle[0] + traced_rectangle[2], traced_rectangle[1] + traced_rectangle[3], (0, 155, 255),
                    2))'''

                cv2.circle(frame, (key_points['left_eye']), 2, (0, 155, 255), 2)
                cv2.circle(frame, (key_points['right_eye']), 2, (0, 155, 255), 2)
                cv2.circle(frame, (key_points['nose']), 2, (0, 155, 255), 2)
                cv2.circle(frame, (key_points['mouth_left']), 2, (0, 155, 255), 2)
                cv2.circle(frame, (key_points['mouth_right']), 2, (0, 155, 255), 2)

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(input_source)
        faces = detector.detect_faces(img)
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        num_faces = len(faces)
        cv2.putText(img, f'Number of faces: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


input_source = input('Enter input source (webcam or image path): ')
detect_faces(input_source)
