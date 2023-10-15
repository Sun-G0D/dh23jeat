import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

import numpy as np 
import numpy as np
import cv2
import os
import json

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

    data = np.load('test1.txt.npy')

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


    bigdata = np.load('bigdata.npy')
    ratings = np.load('ratings.npy')

    print(ratings)

    # This builds the model for the first time:
    bigmodel.fit(bigdata, ratings, batch_size=10, epochs=300, shuffle=True)
    bigmodel.save_weights('bigmodel.h5')
    input = np.array([  [0]])
    print(input.shape)
    print(bigmodel(input))

    @app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (32,32,3))
    
    #Step 3
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      print(probabilities)
      #Step 4
      number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 
'truck']
      index = np.argsort(probabilities)
      predictions = {
        "class1":number_to_class[index[9]],
        "class2":number_to_class[index[8]],
        "class3":number_to_class[index[7]],
        "prob1":probabilities[index[9]],
        "prob2":probabilities[index[8]],
        "prob3":probabilities[index[7]],
      }
    #Step 5
    return render_template('predict.html', predictions=predictions)

app.run(host='0.0.0.0', port=80)