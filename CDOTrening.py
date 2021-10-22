import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import pickle
import time

# wczytywanie z pliku
pickleX = open("PhotosGray.pickle", "rb")
X = pickle.load(pickleX)

pickleY = open("ClassGray.pickle", "rb")
Y = pickle.load(pickleY)

# najbardziej optymalne parametry
dense_layers = [1]
layer_sizes = [32]
conv_layers = [6]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
                conv_layer, layer_size, dense_layer, int(time.time()))  # zapisanie zestawu parametrów
                
            # tworzenie logów do analizy
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            model = Sequential()  # nasz model będzie miał jeden input i jeden output

            # stworzenie z macierzy 3x3 macierzy 2x2
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                # stworzenie z macierzy 3x3 macierzy 2x2
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())  # tworzymy z macierzy wektor
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

model.add(Dense(3))  # 3 ponieważ mamy 3 opcje
model.add(Activation('softmax'))  # zwraca więcej niż dwie opcje
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=['categorical_accuracy'])  # przygotowanie do treningu, categorical_crossentropy ponieważ 3 opcje
model.fit(X, Y, epochs=11, batch_size=32,
          validation_split=0.1, callbacks=[tensorboard])  # trening
model.save("CDO.model")  # zapisanie modelu
