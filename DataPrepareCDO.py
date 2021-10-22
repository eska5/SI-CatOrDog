import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle

training_data = []
Data = "C:/Users/Kuba/Desktop/SemestrIV/Sztuczna Inteligencja/Project/PetImagesGeneral"
Categories = ["Cat", "Dog", "Other"]
imgSize = 200


def create_data():
    for category in Categories:
        path = os.path.join(Data, category)  # path
        classNumber = Categories.index(category)
        for img in os.listdir(path):
            try:
                listOne = [0, 0, 0]
                listOne[classNumber] = 1
                imgArray = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                newArray = cv2.resize(imgArray, (imgSize, imgSize))
                training_data.append([newArray, listOne])
            except Exception as e:
                pass


create_data()

random.shuffle(training_data)

Photos = []
Class = []

for features, label in training_data:
    Photos.append(features)
    Class.append(label)

PhotosGray = np.array(Photos).reshape(-1, imgSize, imgSize, 1)
ClassGray = np.array(Class)

# zapisywanie do pliku
pickle_out = open("PhotosGray.pickle", "wb")
pickle.dump(PhotosGray, pickle_out)
pickle_out.close()

pickle_out = open("ClassGray.pickle", "wb")
pickle.dump(ClassGray, pickle_out)
pickle_out.close()
