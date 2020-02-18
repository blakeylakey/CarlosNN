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

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input], layers, allow_input_downcast=False)
    feature_maps = get_feature([instance])
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(150,150),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction (binary on cats and dogs)")
    parser.add_argument("-i",'--img', help="Path to image for feature extraction")
    parser.add_argument("-o","--output",help="Path to output image")
    args = parser.parse_args()
    img_path = args.img
    output_path = args.output
    if img_path:
        model = build_model()
        model.load_weights('CAMWeights.h5')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='binary_crossentropy')
        im_original = cv2.resize(cv2.imread(img_path), (150,150))
        im = np.expand_dims(im_original, axis=0)
        layers_extract = [3]
        hc = extract_hypercolumn(model, layers_extract, im)
        ave = np.average(hc, axis=2)
        plt.imshow(ave)
        plt.savefig(output_path)
        plt.show()







