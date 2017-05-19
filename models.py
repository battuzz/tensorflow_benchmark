from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, ZeroPadding2D

def build_model_1():
    model= Sequential()
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    model.add(MaxPool2D(2))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64))
    model.add(Flatten()) 
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model

def build_model_2():
    model= Sequential()
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64))
    model.add(MaxPool2D(2))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64))
    model.add(Flatten()) 
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model

def build_model_3():
    model= Sequential()
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2))

    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    model.add(MaxPool2D(2))
    
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64))
    model.add(MaxPool2D(2))
    
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=64))
    
    model.add(Flatten()) 
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model

def build_model_4():
    model= Sequential()
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32, input_shape=(32, 32, 3)))
    model.add(MaxPool2D(2))

    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    model.add(MaxPool2D(2))
    
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    model.add(MaxPool2D(2))

    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    model.add(MaxPool2D(2))
    
    model.add(ZeroPadding2D(padding=(2, 2), data_format=None))
    model.add(Conv2D(kernel_size=5, activation='relu', filters=32))
    
    model.add(Flatten()) 
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model