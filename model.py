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

######################################################################################################################
# Parameter to tune the Training 

EPOCHS = 20
SAMPLE_PER_EPOCHS = 20000

DATA_DIR = 'data'
LOG_DIR = './logs'
MODEL_DIR = './models'

LEARNING_RATE = 1e-4
MIN_DELTA=0.0001

######################################################################################################################

print("Load Data...")
data = loadData(DATA_DIR)
print("Data loaded.")

print("\nSplit data")
data_train, data_val = train_test_split(data, test_size=0.2)
print("Number of train images =", len(data_train))
print("Number of n_val labels =", len(data_val))

VALIDATION_SAMPLES = len(data_val)

model = nvidia_model(False)

print("\nPrepare Training")

gen_train = generator(data_train, augment=True)  
gen_val = generator(data_val)



if os.path.exists(LOG_DIR):
	print('\nRemove logs')
	shutil.rmtree(LOG_DIR)

if not os.path.exists(MODEL_DIR ):
	print('\nCreate model_dir')
	os.mkdir(MODEL_DIR) 

adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

print("\nTrain network ...")

date_string = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
history_object = model.fit_generator(gen_train,
                    steps_per_epoch=SAMPLE_PER_EPOCHS,
                    validation_data=gen_val,
                    validation_steps=VALIDATION_SAMPLES, 
                    epochs=EPOCHS,
                    callbacks=[
				   keras.callbacks.ModelCheckpoint(MODEL_DIR  + '/model.' + date_string + '.{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
                       keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=1, verbose=0, mode='auto'),
                       keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=True, write_images=False)
                    ])
print("\n... Training finished")
