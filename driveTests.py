import sys
import argparse
import base64
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
import helper

import cv2 

activation_relu = 'relu'

# Check if this can be moved to the Class
def create_model(hd5Path):
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

    # starts with five convolutional and maxpooling layers
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.load_weights(hd5Path)

    return model


# Create model and load weights
model = create_model("model.h5")
model._make_predict_function()
graph = tf.get_default_graph()

def main():
    cv_img = cv2.imread('images/1.jpg')
    
    image_array = np.asarray(cv_img)

    image_array = helper.crop(image_array, 0.35, 0.1)
    image_array = helper.resize(image_array, new_dim=(64, 64))

    transformed_image_array = image_array[None, :, :, :]

    print(float(model.predict(transformed_image_array)))


if __name__ == "__main__":
    main()

