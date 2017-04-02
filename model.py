from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
import time
import shutil
import os
import random
import cv2
import math
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

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
from network import *
from generator import *



data_dir = 'data'

print("Load Data...")
data = loadData(data_dir)
print("Data loaded.")

print("\nSplit data")
data_train, data_val = train_test_split(data, test_size=0.2)
print("Number of train images =", len(data_train))
print("Number of n_val labels =", len(data_val))

model = nvidia_model(False)

print("\nPrepare Training")
import shutil


adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

gen_train = generator(data_train, augment=True)  
gen_val = generator(data_val)

log_dir = './logs'
model_dir = './models'
date_string = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
nb_epoch = 20
nb_samples_per_epoch = 8000
nb_val_samples = len(data_val)
learning_rate = 1e-4
min_delta=0.0001

if os.path.exists(log_dir):
	print('\nRemove logs')
	shutil.rmtree(log_dir)

if not os.path.exists(model_dir):
	print('\nCreate model_dir')
	os.mkdir(model_dir) 

print("\nTrain Data")

history_object = model.fit_generator(gen_train,
                    steps_per_epoch=nb_samples_per_epoch,
                    validation_data=gen_val,
                    validation_steps=nb_val_samples, 
                    epochs=nb_epoch,
                    callbacks=[
				   keras.callbacks.ModelCheckpoint(model_dir + '/model.' + date_string + '.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
                       keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=1, verbose=0, mode='auto'),
                       keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
                    ])


model.save("model.h5")
print("\nSaved model")
