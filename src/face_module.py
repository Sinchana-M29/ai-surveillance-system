import cv2
import os
import numpy as np

known_faces = []
known_names = []


def load_known_faces():
    folder = "known_faces"

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        known_faces.append(gray)
        known_names.append(file)


def recognize_face(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    best_match = None
    min_diff = float("inf")

    for i, known_face in enumerate(known_faces):
        known_resized = cv2.resize(known_face, (100, 100))
        test_resized = cv2.resize(face_gray, (100, 100))

        diff = np.sum((known_resized - test_resized) ** 2)

        if diff < min_diff:
            min_diff = diff
            best_match = known_names[i]

    if min_diff < 3000000:
        return best_match
    else:
        return "Unknown"
