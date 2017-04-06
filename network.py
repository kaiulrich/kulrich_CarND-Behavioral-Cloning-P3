import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, Lambda, MaxPooling2D, Cropping2D, AveragePooling2D,BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from utils import *

def nvidia_model(summary=False):
    
    model = Sequential()
    model.add(Cropping2D(((80,25),(1,1)), input_shape=[160, 320, 3], name="Crop2D"))
    model.add(BatchNormalization(axis=1, name="Normalise"))
   
    # Successively learn through multiple convolutions, relu activations and pooling layers,
    model.add(Convolution2D(24, (3, 3), strides=(2,2), name="Conv1", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool1"))
    model.add(Convolution2D(48, (3, 3), strides=(1,1), name="Conv2", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool2"))
    model.add(Convolution2D(72, (3, 3), strides=(1,1), name="Conv3", activation="relu"))
    model.add(MaxPooling2D(name="MaxPool3"))
    model.add(Dropout(0.2, name="Dropout1"))

    model.add(Flatten(name="Flatten"))
    model.add(Dense(100, activation="relu", name="FC2"))
    model.add(Dropout(0.5, name="Dropout2"))
    model.add(Dense(50, activation="relu", name="FC3"))
    model.add(Dropout(0.2, name="Dropout3"))
    model.add(Dense(10, activation="relu", name="FC4"))

    # Final Output  of steering angles
    model.add(Dense(1, name="Steering", activation='linear'))


    if summary:
        model.summary()
    return model

