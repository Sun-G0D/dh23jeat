import numpy as np
import cv2
import os
import json

'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import device_lib
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
print(device_lib.list_local_devices())
'''
import tensorflow as tf

from os import listdir
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(20, (3, 3), activation='relu', padding='same', strides=2)
        ])

        self.decoder = tf.keras.Sequential([
           # layers.Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
            layers.Conv2DTranspose(20, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(512, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':

    autoencoder = Autoencoder()
    adamOpti = tf.keras.optimizers.Adam(learning_rate = 0.0005)
    autoencoder.compile(optimizer=adamOpti, loss='mean_squared_error')


    autoencoder.decoder.build(input_shape=(None, 1, 1, 20))
    #print(autoencoder.encoder.summary())
    #print(autoencoder.decoder.summary())

    data = np.load('test1.txt.npy')

    #data = np.empty(shape = (20136, 64, 64, 3))
    '''
    i = 0
    for path in os.listdir('shoes'):
        if (path.endswith(".jpg")):
            img = cv2.imread('shoes/' + path)
            im = cv2.resize(img, (64, 64))
            data[i] = im/255
            i += 1
            if i % 100 == 0:
                print(i)
    '''

    '''
    print("done")
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        autoencoder.fit(data, data, epochs=1, batch_size=25,validation_split=.15, shuffle=True)
    autoencoder.encoder.save_weights('enc_shoes.h5')
    autoencoder.decoder.save_weights('dec_shoes.h5')
    '''
    test = data[21]
    cv2.imwrite("test2.jpg", test * 255)

    autoencoder.encoder.load_weights('enc_top.h5')
    autoencoder.decoder.load_weights('dec_top.h5')

    pants = tf.keras.models.clone_model(autoencoder)
    pants.decoder.build((None, 1, 1, 20))
    pants.encoder.load_weights('enc_pants.h5')
    pants.decoder.load_weights('dec_pants.h5')
    shoes = tf.keras.models.clone_model(autoencoder)
    shoes.encoder.load_weights('enc_shoes.h5')
    shoes.decoder.build((None, 1, 1, 20))
    shoes.decoder.load_weights('dec_shoes.h5')

    '''
    intest = np.array([test])
    out = pants.call(intest)
    out = np.array(out)
    cv2.imwrite("test.jpg", out[0] * 255)
    '''

    bigmodel = tf.keras.Sequential()
    bigmodel.add(tf.keras.layers.Dense(60, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(512, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(256, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(128, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(64, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(32, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(16, activation='relu'))
    bigmodel.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    bigmodel.compile(optimizer='adam', loss='mse')

    bigdata = np.empty(shape=(333, 60))
    ratings = np.empty(333)
    j = 0

    '''

    with open('train_no_dup.json') as f:
        d = json.load(f)
        for fit in d:
            setid = fit["set_id"]
            path = "C:/Users/evans/Downloads/polyvore-images/images/" + setid
            if (not os.path.exists(path)):
                continue
            clothes = fit["items"]
            hit = [False, False, False]
            vector = np.empty(60)
            for item in clothes:
                if item["categoryid"] == 11:
                    hit[0] = True
                    img = cv2.imread(path + "/" + str(item["index"]) + ".jpg")
                    topIm = cv2.resize(img, (64, 64))/255
                    vector[0:20] = autoencoder.encoder(np.array([topIm]))
                    #top
                if item["categoryid"] == 28:
                    hit[1] = True
                    img = cv2.imread(path + "/" + str(item["index"]) + ".jpg")
                    pantsIm = cv2.resize(img, (64, 64))/255
                    vector[20:40] = pants.encoder(np.array([pantsIm]))
                    #pants
                if item["categoryid"] == 41:
                    hit[2] = True
                    img = cv2.imread(path + "/" + str(item["index"]) + ".jpg")
                    shoesIm = cv2.resize(img, (64, 64))/255
                    vector[40:60] = shoes.encoder(np.array([shoesIm]))
                    #shoes
            if hit[0] and hit[1]:
                bigdata[j] = vector
                ratings[j] = fit["likes"]/fit["views"]
                j += 1
                print(j)

    np.save('bigdata.npy', bigdata)
    np.save('ratings.npy', ratings)

    '''

    bigdata = np.load('bigdata.npy')
    ratings = np.load('ratings.npy')

    print(ratings)

    # This builds the model for the first time:
    bigmodel.fit(bigdata, ratings, batch_size=10, epochs=300, shuffle=True)
    bigmodel.save_weights('bigmodel.h5')
    input = np.array([  [0]])
    print(input.shape)
    print(bigmodel(input))