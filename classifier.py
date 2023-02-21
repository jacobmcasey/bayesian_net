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
    def __init__(self, layers = [25,8,4], learning_rate=0.001, iterations=100):
        """
        Initializes a Classifier object with the specified hyper-parameters.

        Args:
        layers (list of int): The number of units in each layer of the neural network.
        learning_rate (float): The learning rate used for weight updates during training.
        iterations (int): The number of iterations used for training the neural network.
        """
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
        """
        Initializes the weights of the neural network with random values close to 0.
        """
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) / 1000
        self.params['b1'] = np.random.randn(self.layers[1],) / 1000
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) / 1000
        self.params['b2'] = np.random.randn(self.layers[2],) / 1000

    def relu(self,Z):
        """
        Applies the ReLU activation function.

        Args:
        Z (numpy.ndarray): Input to which the ReLU function will be applied.

        Returns:
        numpy.ndarray: Result of the ReLU function applied to the input.
        """
        return np.maximum(0,Z)

    def dRelu(self, x):
        """
        Calculate output when given to the derivative of the ReLU function.

        Args:
        x (numpy.ndarray): Input array.

        Returns:
        numpy.ndarray: Result of the inverse ReLU function applied to the input.
        """
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def softmax(self, x):
        """
        Compute the softmax of the input (used for final layer of multi-class problem).

        Args:
        x (ndarray): The input array.

        Returns:
        ndarray: The softmax of the input array.
        """
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def eta(self, x):
        """
        Computes small value eta (1e-10), to avoid division by zero/NaNs.
    
        Args:
        x (numpy array): Input array/vector.
    
        Returns:
        numpy array: Either x or eta value, depending which is bigger
        """
        eta = 0.000000001
        return np.maximum(x, eta)
    
    def categorical_cross_entropy_loss(self, y, yhat):
        # Calculate categorical cross entropy
        nsample = y.shape[0]
        yhat = self.eta(yhat)
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

    def likelihood(self, p, data):
        number_feat = len(p)
        p_xi_y = []
        for m in range(number_feat):
            p_i_y = (p[m] ** data[m]) * ((1 - p[m]) ** (1 - data[m]))
            p_xi_y.append(p_i_y)
        p_xi_y_ = [1 if x == 0 else x for x in p_xi_y]
        likelihood = np.prod(p_xi_y_)
        return likelihood

    def posteriori(self, l0, pi0, l1, pi1, l2, pi2, l3, pi3):
        post0 = l0 * pi0
        post1 = l1 * pi1
        post2 = l2 * pi2
        post3 = l3 * pi3
        return [post0, post1, post2, post3]
    
    def argmax(self, post):
        ind = post.index(max(post))
        return ind

    def reset(self):
        self.params = {}
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

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

        # Comment to return and bypass bernoulli 
        # return np.argmax(pred, axis=1)

        p_y_0, p_y_1, p_y_2, p_y_3 = pred[0]

        p_i_y_0 = [0.0, 0.7857142857142857, 0.21428571428571427, 0.6428571428571429, 0.7142857142857143, 0.10714285714285714, 0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p_i_y_1 = [0.7575757575757576,0.0, 0.8484848484848485, 0.12121212121212122, 0.06060606060606061, 0.42424242424242425, 0.06060606060606061, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p_i_y_2 = [0.19230769230769232, 0.6538461538461539, 0.0, 0.7307692307692307, 0.038461538461538464, 0.11538461538461539, 0.8461538461538461, 0.19230769230769232, 0.0, 0.0, 0.0, 0.0, 0.0, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.038461538461538464]
        p_i_y_3 = [0.9696969696969697, 0.18181818181818182, 0.8484848484848485, 0.0, 0.0, 0.030303030303030304, 0.06060606060606061, 0.9090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        l0 = self.likelihood(p_i_y_0, state_data)
        l1 = self.likelihood(p_i_y_1, state_data)
        l2 = self.likelihood(p_i_y_2, state_data)
        l3 = self.likelihood(p_i_y_3, state_data)
        post = self.posteriori(l0, p_y_0, l1, p_y_1, l2, p_y_2, l3, p_y_3)
        class_ = self.argmax(post)

        return class_
