import numpy as np
import math


class NeuralNetwork:
    def __init__(self, learningRate):
        self.learningRate = learningRate

        self.weights1 = np.array([np.random.randn(), np.random.randn()])
        self.bias1 = np.random.randn()

        self.weights2 = np.array([np.random.randn(), np.random.randn()])
        self.bias2 = np.random.randn()
        
        self.weights3 = np.array([np.random.randn(), np.random.randn()])
        self.bias3 = np.random.randn()


    #Rectified Linear Unit Activation function f(x)=max(0,x)
    def relu(x):
        return np.max(0,x)

    def relu_derivative(x):
        return (x > 0).astype(float)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self,x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    #Converting inputs to Linear Regresion y = ax + b -> Z = X @ W + B 
    def forward(self, X): 
        self.Z1 = X @ self.weights1 + self.bias1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.weights2 + self.bias2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.weights3 + self.bias3 
        self.A3 = self.sigmoid(self.Z3)  

        return self.A3 





        
        
        
        