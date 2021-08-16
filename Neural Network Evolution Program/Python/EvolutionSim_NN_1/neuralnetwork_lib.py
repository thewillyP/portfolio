import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class NN_Manager:

    def __init__(self, dat, label, tenslan, parse):
        self.data = dat
        self.labels = label
        self.tensorlabel = tenslan
        self.label_number = len(self.tensorlabel[0])
        self.training_data, self.test_data = np.split(self.data,[parse])
        self.features = len(self.training_data[0])
        # Create normalize parameters
        mu = np.mean(self.training_data, axis=0)
        sigma = np.std(self.training_data - mu, axis=0)
        # normalize test and train data sets
        self.training_data = self.normalize(self.training_data, mu, sigma)
        self.test_data = self.normalize(self.test_data, mu, sigma)

        self.training_label, self.test_label = np.split(self.labels,[parse])


    def train(self, pop):
        lamk = pop._components[0].val
        lamb = pop._components[1].val
        iter = pop._components[2].val
        layer_max = pop._components[3].val
        arch = pop._components[3].neuronArray


        model = keras.Sequential()
        # Create Input shape
        # Number of neurons, number of previous neurons, lamk, lamb
        model.add(keras.layers.Dense(arch[0], input_dim=self.features, kernel_regularizer=keras.regularizers.l2(lamk),
                                     bias_regularizer=keras.regularizers.l2(lamb), activation="relu"))

        for i in range(1,layer_max):
            model.add(keras.layers.Dense(arch[i], kernel_regularizer=keras.regularizers.l2(lamk),
                                         bias_regularizer=keras.regularizers.l2(lamb), activation="relu"))

        model.add(keras.layers.Dense(self.label_number, activation="sigmoid"))

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        model.fit(self.training_data, self.training_label, epochs=iter, verbose=0)
        test_loss, test_acc = model.evaluate(self.test_data, self.test_label)
        return test_acc


    def normalize(self, X, mu, sigma):
        return np.divide((X - mu), sigma)




        



