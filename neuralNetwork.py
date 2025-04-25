import numpy as np
import math


class NeuralNetwork:
    def __init__(self, learningRate):
        self.learningRate = learningRate

        self.weights1 = np.random.randn(12288, 128) #12288 inputs and 128 neurons
        self.bias1 = np.random.randn(1, 128)

        self.weights2 = np.random.randn(128, 64)
        self.bias2 = np.random.randn(1, 64)

        self.weights3 = np.random.randn(64, 3)
        self.bias3 = np.random.randn(1, 3)


    #Rectified Linear Unit Activation function f(x)=max(0,x)
    def relu(self, x):
        return np.max(0,x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    #Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    #Converting inputs to Linear Regresion y = ax + b -> Z = X * W + B
    #Activation function A = RELU(Z)
    def forward(self, X): 
        self.Z1 = X @ self.weights1 + self.bias1
        self.A1 = self.relu(self.Z1)

        self.Z2 = self.A1 @ self.weights2 + self.bias2
        self.A2 = self.relu(self.Z2)

        self.Z3 = self.A2 @ self.weights3 + self.bias3 
        self.A3 = self.sigmoid(self.Z3)  

        return self.A3 

    def backward(self, X, Y):
        dA3 = self.A3 - Y
        dZ3 = dA3 * self.sigmoid_derivative(self.Z3)

        #TO DO: complete this back propagation function, go back to each layer and calculate error 


    def update_parameters(self, weight_error, bias_error):
        return