import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

from os import listdir
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(496, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(248, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(124, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(124, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(248, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(496, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    folder_dir = "C:/Users/jeffr_qv7a35e/Downloads/Re-PolyVore/Re-PolyVore/top"
    a = np.empty(shape = (19397, 64, 64, 3))
    """
    i = 0
    for images in os.listdir("C:/test"):
        img = cv2.imread("C:/test/" + images)
        a[i] = img/255
        i += 1

    np.save('test1', a)
    
    print("done")
    """
    d = np.load('test1.txt.npy')
    autoencoder.decoder.build(input_shape=(None, 1, 1, 16))
    """
    for images in os.listdir(folder_dir):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            img = cv2.imread(folder_dir + "/" + images)
            im = cv2.resize(img, (64, 64))
            cv2.imwrite("C:/test/" + images, im)
    """

    autoencoder.fit(d, d, epochs=400, batch_size=100, validation_split=.15, shuffle=True)