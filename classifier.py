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
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self, layers=[25,8,4], learning_rate=0.001, iterations=100):
        # Define hyper-parameters
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None

    def init_weights(self):
        # Initialise random weights
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
    
    def relu(self,Z):
        # Relu function 
        return np.maximum(0,Z)

    def dRelu(self, x):
        # Derivative of relu
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def softmax(self, x):
        # Softmax
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def eta(self, x):
        # Minimum value eta
        ETA = 0.0000000001
        return np.maximum(x, ETA)
    
    def categorical_cross_entropy_loss(self, y, yhat):
        # Calculate categorical cross entropy
        nsample = y.shape[0]
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        loss = -1/nsample * np.sum(y * np.log(yhat))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        Returns output and loss
        '''
        
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.softmax(Z2)
        loss = self.categorical_cross_entropy_loss(self.y,yhat)

        # save calculated parameters     
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss
        
    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        Feed the result backwards!
        '''
        dl_wrt_yhat = yhat - self.y # derivative of loss w.r.t yhat
        
        dl_wrt_z2 = dl_wrt_yhat
        
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
        
        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)
        
        # Update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def int_to_onehot(self, int_list):
        num_classes = max(int_list) + 1
        onehot = np.eye(num_classes)[int_list]
        return onehot

    def reset(self):
        self.model = None

    def fit(self, good_moves_data, target):
        '''
        Trains the neural network using the specified data and labels
        '''

        print (target)
        self.X = np.array(good_moves_data)
        self.y = self.int_to_onehot(target)
        self.init_weights() #initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)
            print("Training loss at iteration", i, ":", loss)

    def predict(self, state_data, legal):
        '''
        Predicts on a test data
        Simply use the weights and biases calculated before
        '''

        state_data = np.array(state_data)
        Z1 = state_data.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.softmax(Z2)
        sum_pred = np.sum(pred, axis=1)
        print("values: ", pred,"with probability sum: ", sum_pred)
        return np.argmax(pred, axis=1)
