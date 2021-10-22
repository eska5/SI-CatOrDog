import numpy as np
import os
import cv2
import random
import pickle
training_data = []
Data = "D:\Politechnika\SEM4\SZTUCZNA INTELIGENCJA\projekt\development\PetImages"
Categories = ["Cats", "Hiro"]
imgSize = 200


def create_training_data():
    for category in Categories:
        path = os.path.join(Data, category)
        for img in os.listdir(path):
            try:
                if category == "Cats":
                    listOne = 1
                else:
                    listOne = 0
                imgArray = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                training_data.append([newArray, listOne])
            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)

X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

Xgray = np.array(X).reshape(-1, imgSize, imgSize, 1) # Jeśli chcemy RGB to zmieniamy ostatni parametr na 3
Ygray = np.array(Y)


# zapisywanie do pliku
pickle_out = open("Xgray.pickle", "wb")
pickle.dump(Xgray, pickle_out)
pickle_out.close()

pickle_out = open("Ygray.pickle", "wb")
pickle.dump(Ygray, pickle_out)
pickle_out.close()
