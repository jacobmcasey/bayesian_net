# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

class Classifier:
    def __init__(self):
        self.model = None

    def reset(self):
        self.model = None

    def fit(self, good_moves_data, target):

        # Convert to one-hot encoded features
        target_encoded = to_categorical(target, num_classes=4)

        # Define the network architecture
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(25,)))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32))
        self.model.add(tf.keras.layers.Dense(4, activation='softmax'))
        self.model.summary()
        
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train the model
        self.model.fit(np.array(good_moves_data), target_encoded, epochs=10, batch_size=None)

    def predict(self, state_data, legal):
        move = self.model.predict(np.array(state_data).reshape(1, 25))
        return np.argmax(move)
