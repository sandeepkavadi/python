# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 23:00:12 2020

@author: sande
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    """Creating a class of perceptrons (linear classifiers)
    The labels have to b either +1 or -1
    Data has to be linearly separable for finding a
    hyperplane that separates the two classes
    To handle cases where the algortihm does not converge,
    we set a maximum number of epochs that the algorithm loops 
    around the data set
    
    Hyper parameters (set by the user)
    ------------------------------------------------------------
    eta - a number between 0.0 and 1.0
    Learning rate - which determines the size of the update made to the
    weight vector
    
    epoch - an integer 
    the number of iterations around the data set, essentially to handle
    the degenerate case when the algorithm does not converge
    
    seed - an integer
    the seed to the random number generator primarily to enable replication
    of results. The random numbers are used to randomly set the 
    weights of the initial vector
    
    Attributes (internal parameters)
    --------------------------------------------------------------
    w - the weight vector
    Essentially the vector used to determine the hyperplane separating
    the data. The weight vector is updated iteratively till there are no
    misclassifications in the case of linearly separable data
    This vector includes a bias weight w[0]
    
    error - the number of misclassifications in the data"""
    
    def __init__(self, eta = 0.01, epochs = 50, seed = 50):
        """Set the hyper parameters for the perceptron"""
        self.eta = eta
        self.epochs = epochs
        self.seed = seed
        
    
    def fit(self, X, y):
        """X is the input data set normalized such that the largest
        norms <= 1.0
        y is the vector of all corresponding labels"""
        
        #Initializa the weight vector to random weights
        
        rgen = np.random.RandomState(self.seed)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.errors = []
        
        for i in range(self.epochs):
            error = 0
            for xi, label in zip(X,y):
                delta_w = self.eta * (label - self.predict(xi))
                self.w[1:] += delta_w * xi #updating the weights
                self.w[0] += delta_w
                error += int(delta_w != 0.0)
            self.errors.append(error)
        return self

    def raw_prod(self, X):
        """Calculates raw output before the linear threshold is 
        applied"""
        return np.dot(X, self.w[1:]) + self.w[0]
      
    def predict(self, X):
        """The decision rule, the raw output computation along with the
        Linear threshold unit to predict the label"""
        return np.where(self.raw_prod(X) >= 0.0, 1, -1)
   
#-------------------------------------------------------------------                
#Example of implementing a perceptronon the iris dataset

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None, 
    encoding='utf-8')       
        
df.head()

y = df.iloc[:,4].values
y = np.where(y=='Iris-setosa', 1, -1)
print(y)



X = df.iloc[:, [0,2]].values

print(X)



slp=Perceptron(eta=0.2, epochs=20, seed=50)

slp.fit(X, y)
plt.plot(range(1, len(slp.errors) + 1), slp.errors)
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.show()




        
    
    
    