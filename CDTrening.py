import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

accThatWeWant = 0.7


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') is not None and logs.get('val_acc') >= accThatWeWant:
            print("70%")
            self.model.stop_training = True


helpfulVar = myCallback()

Program = "CezarPsy{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(Program))

# wczytywanie z pliku
pickleX = open("PhotosGray.pickle", "rb")
X = pickle.load(pickleX)

pickleY = open("ClassGray.pickle", "rb")
Y = pickle.load(pickleY)

X = X/255.0

# najbardziej optymalne parametry
dense_layers = 0
layer_size = 32
conv_layer = 2

NAME = "{}-conv-{}-nodes-{}-dense-{}".format(
    conv_layer, layer_size, dense_layers, int(time.time())) # zapisanie zestawu parametrów

model = Sequential()# nasz model będzie miał jeden input i jeden output
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
model.add(Dense(layer_size))
model.add(Activation('relu'))
model.add(Dense(1))  # 1 ponieważ mamy dwie opcje
model.add(Activation('sigmoid'))  # sigmoid zwraca 1 albo 0
tensorboard = TensorBoard(
    log_dir="finalllogs/{}".format("final_test"))  # zapisanie logów
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'],)  # przygotowanie do treningu, binary_crossentropy ponieważ 2 opcje
model.fit(X, Y, batch_size=32, epochs=9,
          validation_split=0.3, callbacks=[tensorboard])  # trening
model.save("FINALLLLBinaryDogCezar.model")  # zapisanie modelu
