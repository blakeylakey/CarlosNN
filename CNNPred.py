from keras.preprocessing import image as image_utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn import cluster
import argparse

import scipy as sp
import keras
import numpy as np
import argparse
import cv2
import h5py
import time

size = 150
weight_path = 'CAMWeights.h5'

def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(150,150,3), name='layer_0'))
    model.add(Activation('relu', name='layer_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='layer_2'))

    model.add(Convolution2D(32, 3, 3, name='layer_3'))
    model.add(Activation('relu', name='layer_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='layer_5'))

    model.add(Convolution2D(64, 3, 3, name='layer_6'))
    model.add(Activation('relu', name='layer_7'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='layer_8'))

    model.add(Flatten(name='layer_9'))  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, name='layer_10'))
    model.add(Activation('relu', name='layer_11'))
    model.add(Dropout(0.5, name='layer_12'))
    model.add(Dense(1, name='layer_13'))
    model.add(Activation('sigmoid', name='layer_14'))

    return model

def predict_img(img_path, weight_path):
    img = image_utils.load_img(img_path, target_size=(size,size))
    img = image_utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    model = build_model()
    model.load_weights(weight_path)

    if model.predict(img)[0][0] == 0:
        return 'cat'
    else:
        return 'dog'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Prediction (binary on cats and dogs)")
    parser.add_argument("-i",'--img', help="Path to image for feature extraction")
    args = parser.parse_args()
    img_path = args.img
    weight_path = 'CAMWeights.h5'
    print(predict_img(img_path, weight_path))
