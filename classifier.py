"""
This module implements a hybrid approach classifier that uses Bernoulli Naive Bayes prediction utilising
priors computed by a 3-layer neural network in NumPy.

Classes:
    Classifier: A hybrid approach classifier that uses Bernoulli Naive Bayes prediction utilising
    priors computed by a 3-layer neural network.

Functions:
    int_to_onehot(int_list): Converts list of integers to a one-hot encoded numpy array.
    relu(Z): Applies the ReLU activation function to the input.
    dRelu(x): Calculate output when given to the derivative of the ReLU function (for backprop).
    softmax(x): Computes the softmax of the input (used for final layer of multi-class problem).
    eta(x): Computes small value eta (1e-10), to avoid division by zero/NaNs.
    categorical_cross_entropy_loss(y, yhat): Calculates the categorical cross-entropy loss between the true labels and the predicted labels.
    forward_propagation(): Performs forward propagation through the network and calculates the loss.
    back_propagation(yhat): Computes the derivatives and updates learnable parameters using this.
    likelihood(p, data): Calculates the likelihood function.
    posteriori(l0, pi0, l1, pi1, l2, pi2, l3, pi3): Compute the posteriori probabilities of each class given the likelihood and prior probabilities.
    argmax(post): Finds the maximum posteriori and returns it's index.
    reset(): Resets the parameters and losses.

"""

import numpy as np

class Classifier:
    """
    Hybrid approach classifier that uses Bernoulli Naive Bayes prediction utilising
    priors computed by a 3-layer neural network.
    """
    def __init__(self, layers = [25,8,4], learning_rate=0.005, iterations=100):
        """
        Initializes a Classifier object with the specified hyper-parameters.
        Args:
        layers (list of int): The number of units in each layer of the neural network.
        learning_rate (float): The learning rate used for weight updates during training.
        iterations (int): The number of iterations used for training the neural network.
        """
        # Define parameters and hyper-parameters
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.layers = layers
        self.params = {}
        self.loss = []
        self.X = None
        self.y = None

    def init_weights(self):
        """
        Initializes the weights of the neural network with random values close to 0.
        """
        np.random.seed(1)
        self.params["weights1"] = np.random.randn(self.layers[0], self.layers[1]) / 1000
        self.params['biases1'] = np.random.randn(self.layers[1],) / 1000
        self.params['weights2'] = np.random.randn(self.layers[1],self.layers[2]) / 1000
        self.params['biases2'] = np.random.randn(self.layers[2],) / 1000

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
        Calculate output when given to the derivative of the ReLU function (for backprop).
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
        """
        Calculates the categorical cross-entropy loss between the true labels and the predicted labels.
        """
        nsample = y.shape[0]
        yhat = self.eta(yhat)
        loss = -1/nsample * np.sum(y * np.log(yhat))
        return loss

    def forward_propagation(self):
        """
        Performs forward propagation through the network and calculates the loss.

        Returns:
        yhat: The output of the network after the final layer, which is the softmax activation.
        loss: The categorical cross-entropy loss calculated between yhat and the true labels y.

        The function calculates the weighted sum of inputs for the first layer and passes it through a ReLU.
        The output is then passed through the second layer and softmax activation. The categorical cross-entropy loss is then
        calculated between yhat and the true labels y. The coefficients for each layer are stored in self.params.
        """       
        Z1 = self.X.dot(self.params['weights1']) + self.params['biases1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['weights2']) + self.params['biases2']
        yhat = self.softmax(Z2)
        loss = self.categorical_cross_entropy_loss(self.y,yhat)

        # store coefficients for each layer  
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat,loss
        
    def back_propagation(self, yhat):
        """
        Computes the derivatives and updates learnable parameters using this.
        Performs backpropagation to calculate the gradients of the loss w.r.t to each parameter
        and update them using gradient descent.
        """

        # Compute derivatives with chain rule 
        dL_dyhat = yhat - self.y # derivative of loss w.r.t yhat
        dL_dz2 = dL_dyhat
        dL_dweights2 = self.params['A1'].T.dot(dL_dz2)
        dL_dbiases2 = np.sum(dL_dz2, axis=0, keepdims=True)     
        dL_dA1 = dL_dz2.dot(self.params['weights2'].T)
        dL_dz1 = dL_dA1 * self.dRelu(self.params['Z1'])
        dL_dweights1 = self.X.T.dot(dL_dz1)
        dL_dbiases1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # Update using gradient descent
        self.params['weights1'] = self.params['weights1'] - self.learning_rate * dL_dweights1
        self.params['weights2'] = self.params['weights2'] - self.learning_rate * dL_dweights2
        self.params['biases1'] = self.params['biases1'] - self.learning_rate * dL_dbiases1
        self.params['biases2'] = self.params['biases2'] - self.learning_rate * dL_dbiases2

    def int_to_onehot(self, int_list):
        """
        Converts list of integers to a one-hot encoded numpy array.
        Args:
        int_list (list): A list of integers to be converted to one-hot encoded format.
        Returns:
        onehot (numpy array): A numpy array of shape where num_classes is the number of unique integers in int_list. Each row of the array is a one-hot encoded representation of the corresponding integer in int_list.
        """
        num_classes = max(int_list) + 1
        onehot = np.eye(num_classes)[int_list]
        return onehot

    #Likelihood:
    #P(X|y)=P(x1|y)P(x2|y)...
    #BERNOULLI: P(xi|y)=P(i|y)^xi * (1-P(i|y))^(1-xi)
    #where y ={0,1,2,3}

    #e.g.P(x1=1|y)=P(1|y)^1 * (1-P(1|y))^(1-1)
    #as a reuslt we need to calculate only P(xi=1|y) for each record and for each class y

    #from the a feature vector in the form of an array of 1s and 0s like this:
    #data[0] = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #Considered - each record to be a feature of the feature vector state (x1= data[0][0] is a record)

    #likelihood function parameters: list of probabilities for a given class e.g [p(x1=1|y=0),p(x2=1|y=0),...,p(xn=1|y=0)] and a new instance (data)
    #using the BERNOULLI formula calculates the probabilities of a new instance values given the class
    #then multiplies them to obtain the likelihood
    
    def likelihood(self, p, data):
        number_feat = len(p)                           #number of features
        p_xi_y = []                                    #is the set of prbabilites {P(x1|y),P(x2|y)...} calculated using #BERNOULLI: P(xi|y)=P(i|y)^xi * (1-P(i|y))^(1-xi)     
        for m in range(number_feat):
            p_i_y = (p[m] ** data[m]) * ((1 - p[m]) ** (1 - data[m]))
            p_xi_y.append(p_i_y)

        #overcome the zero frequency problem before multipling the probabilites of a feature value for a given class 
        #(any number multiplied by 1 is itself)
        #when the probability is zero replace with one         
        p_xi_y_ = [1 if x == 0 else x for x in p_xi_y]

        likelihood = np.prod(p_xi_y_)
        return likelihood

    def posteriori(self, l0, pi0, l1, pi1, l2, pi2, l3, pi3):
        """
        Compute the posteriori probabilities of each class given the likelihood and prior probabilities.
        Args:
            l0 (float): Likelihood of the data given class 0.
            pi0 (float): Prior probability of class 0.
            l1 (float): Likelihood of the data given class 1.
            pi1 (float): Prior probability of class 1.
            l2 (float): Likelihood of the data given class 2.
            pi2 (float): Prior probability of class 2.
            l3 (float): Likelihood of the data given class 3.
            pi3 (float): Prior probability of class 3.
        Returns:
        list: A list of posteriori probabilities [p(y=0|X),p(y=1|X),p(y=2|X),p(y=3|X)].
        """
        post0 = l0 * pi0
        post1 = l1 * pi1
        post2 = l2 * pi2
        post3 = l3 * pi3
        return [post0, post1, post2, post3]

    def argmax(self, post):
        """
        Finds the maximum posteriori and returns it's index where the index coresponds to the class
        Args: 
        post(list): list of posteriors
        """
        ind = post.index(max(post))
        return ind

    def reset(self):
        self.params = {}
        self.loss = []
        self.X = None
        self.y = None

    def fit(self, good_moves_data, target):
        """
        Trains the neural network weights/biases, with training data and target values

        Args:
        good_moves_data (list): A list of the input features.
        target (list): A list of corresponding target values.
        """
        self.X = np.array(good_moves_data)
        self.y = self.int_to_onehot(target) # Convert moves to one hot encoded
        self.init_weights()

        for _ in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, state_data, legal):
        '''
        Predicts, using a given input state of pacman (assuming medium-grid layout w/ 25 features). 
        Give the last layer of the neural network to the naive bayes approach

        Returns:
        best_move: The move that pacman should take given state_data
        '''

        state_data = np.array(state_data)
        Z1 = state_data.dot(self.params['weights1']) + self.params['biases1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['weights2']) + self.params['biases2']
        nn_predictions = self.softmax(Z2)
        
        # Comment out to return and bypass bernoulli & use neural network last layer
        # return np.argmax(pred, axis=1)

        # This part of the code uses bernoulli naive bayes concept to predict the label. The GOAL is to calculate the probabilities for each possible target y {0,1,2,3}, and pick the class with the highest posteriori
        
        #PRIORS = weights from NN last layer
        #p_y_0 = P(y = 0) 

        p_y_0, p_y_1, p_y_2, p_y_3 = nn_predictions[0]

        #p_i_y_0 = probabilityfeat(0) calculated using probabilityfeat function applied to the training data set (goodmoves.txt) see the function bellow

        p_i_y_0 = [0.0, 0.7857142857142857, 0.21428571428571427, 0.6428571428571429, 0.7142857142857143, 0.10714285714285714, 0.0, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p_i_y_1 = [0.7575757575757576,0.0, 0.8484848484848485, 0.12121212121212122, 0.06060606060606061, 0.42424242424242425, 0.06060606060606061, 0.09090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p_i_y_2 = [0.19230769230769232, 0.6538461538461539, 0.0, 0.7307692307692307, 0.038461538461538464, 0.11538461538461539, 0.8461538461538461, 0.19230769230769232, 0.0, 0.0, 0.0, 0.0, 0.0, 0.038461538461538464, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.038461538461538464]
        p_i_y_3 = [0.9696969696969697, 0.18181818181818182, 0.8484848484848485, 0.0, 0.0, 0.030303030303030304, 0.06060606060606061, 0.9090909090909091, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.030303030303030304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    
        # likelihood function takes the list of probabilities for a given class (e.g.p_i_y_0) and a new instance 
        # calculates the P(X|y)=P(x1|y)P(x2|y).. using BERNOULLI: P(xi|y)=P(i|y)^xi * (1-P(i|y))^(1-xi)
  
        l0 = self.likelihood(p_i_y_0, state_data)
        l1 = self.likelihood(p_i_y_1, state_data)
        l2 = self.likelihood(p_i_y_2, state_data)
        l3 = self.likelihood(p_i_y_3, state_data)
       
        post = self.posteriori(l0, p_y_0, l1, p_y_1, l2, p_y_2, l3, p_y_3)
        best_move = self.argmax(post)

        return best_move

### TRAINING CODE ###
#FUNCTION probabilityfeat(cl) #- used during training of naive bayes distribution
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#the function calculates P(xi=1|y) for each given class y={0,1,2,3}
#parameters:
#cl - the class for which we want to return the probability
#return:
#according to what class we want it returns a list of probabilities e.g.[p(x1=1|y=0),p(x2=1|y=0),...,p(xn=1|y=0)]
#
#import pandas as pd 
# def probabilityfeat(cl):
#     r=pd.DataFrame(data)
                                           
#     prob_xi_1_class0=[]  #P(xi=1|y=0) for each x1 x2 x3...  (where x1 x2 x3 ... are columns in the goodmoves.txt)
#     prob_xi_1_class1=[]  #P(xi=1|y=1) for each x1 x2 x3...
#     prob_xi_1_class2=[]  #P(xi=1|y=2) for each x1 x2 x3...
#     prob_xi_1_class3=[]

#     total_instances=r.shape[0]  #number of instances
    
# #indexes of intances that are in each class
#     inx0=[]   #indexes of instances that are in class 0
#     inx1=[]   #indexes of instances that are in class 1
#     inx2=[]   #indexes of instances that are in class 2
#     inx3=[]   #indexes of instances that are in class 3
    
#     for i in range(total_instances):
#         if target[i]==0 :
#             inx0.append(i)
#         if target[i]==1 :
#             inx1.append(i)
#         if target[i]==2 :
#             inx2.append(i)
#         if target[i]==3 :
#             inx3.append(i)   
    
# #number of instances in each class   
#     inx0len=len(inx0)
#     inx1len=len(inx1)
#     inx2len=len(inx2)
#     inx3len=len(inx3)

# #features in each class
#     class0_instances=r.iloc[inx0]
#     class1_instances=r.iloc[inx1]
#     class2_instances=r.iloc[inx2]
#     class3_instances=r.iloc[inx3]
# 

# #length of a feature vector
#     feat=len( r.iloc[0])

#     one_per_feat_class0=[ ] #the number of ones for each feature/record in the feature vector class 0 instances
#     one_per_feat_class1=[ ] #the number of ones for each feature/record in the feature vector class 1 instances
#     one_per_feat_class2=[ ] #the number of ones for each feature/record in the feature vector class 2 instances
#     one_per_feat_class3=[ ] #the number of ones for each feature/record in the feature vector class 3 instances

# 
# #class 0
# #P(xi=1|y=0) for each record (x1,x2...)
#     for k in range(feat):
#         one=(class0_instances[k] == 1).sum()   #number of instances for each record x1,x2... that are equal to 1 and have class 0
#         one_per_feat_class0.append(one)        
    
# # number of class 0 instances that are equal to 1 for each record /the number of instances in class 0
#     for i in range(feat):
#         pr=one_per_feat_class0[i]/ inx0len   
#         prob_xi_1_class0.append(pr)
    
#  #class 1
# #P(xi=1|y=1) for each record (x1,x2...)
#     for k in range(feat):
#         one=(class1_instances[k] == 1).sum()   
#         one_per_feat_class1.append(one)        
    
#     for i in range(feat):
#         pr=one_per_feat_class1[i]/ inx1len
#         prob_xi_1_class1.append(pr)
    
#  #class 2
# #P(xi=1|y=2) for each record (x1,x2...)
#     for k in range(feat):
#         one=(class2_instances[k] == 1).sum()   
#         one_per_feat_class2.append(one)        
    
#     for i in range(feat):
#         pr=one_per_feat_class2[i]/ inx2len 
#         prob_xi_1_class2.append(pr)    
    
#  #class 3
# #P(xi=1|y=3) for each record (x1,x2...)
#     for k in range(feat):
#         one=(class3_instances[k] == 1).sum()   
#         one_per_feat_class3.append(one)        
    
#     for i in range(feat):
#         pr=one_per_feat_class3[i]/ inx3len 
#         prob_xi_1_class3.append(pr)
      
#     if  cl==0 :
#         return prob_xi_1_class0
#     elif cl==1:
#         return prob_xi_1_class1
#     elif cl==2:
#         return  prob_xi_1_class2
#     elif cl==3:
#         return prob_xi_1_class3
#----------------------------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------------------------
