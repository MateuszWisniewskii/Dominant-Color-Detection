import numpy as np
import math


class NeuralNetwork:
    def __init__(self, learningRate):
        self.learningRate = learningRate

        #He weights initialization for ReLU
        self.weights1 = np.random.randn(12288, 128) * np.sqrt(2. / 12288)
        self.bias1 = np.zeros((1, 128))

        #He weights initialization for ReLU
        self.weights2 = np.random.randn(128, 64) * np.sqrt(2. / 128)
        self.bias2 = np.zeros((1, 64))

        #Xavier initialization for Sigmoid
        self.weights3 = np.random.randn(64, 3) * np.sqrt(1. / 64)
        self.bias3 = np.zeros((1, 3))


    #Rectified Linear Unit Activation function f(x)=max(0,x)
    def relu(self, x):
        return np.maximum(0,x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    #Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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

        #for more than one input
        batch = X.shape[0]

        dA3 = self.A3 - Y
        dZ3 = dA3 * self.sigmoid_derivative(self.Z3)

        dA2 = dZ3 @ self.weights3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)

        dA1 = dZ2 @ self.weights2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        #Gradient for weights and biases
        dW3 = self.A2.T @ dZ3 / batch
        db3 = np.sum(dZ3, axis=0, keepdims=True) / batch

        dW2 = self.A1.T @ dZ2 / batch
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch

        dW1 = X.T @ dZ1 / batch
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch

        self.update_parameters(dW1, db1, dW2, db2, dW3, db3)




    #Method for updating parameters based on calculated gradient
    def update_parameters(self, dW1, db1, dW2, db2, dW3, db3):
        self.weights1 -= self.learningRate * dW1
        self.bias1 -= self.learningRate * db1

        self.weights2 -= self.learningRate * dW2
        self.bias2 -= self.learningRate * db2

        self.weights3 -= self.learningRate * dW3
        self.bias3 -= self.learningRate * db3



    # X - input
    # Y - target
    # epochs - number of iterationss
    def train(self, X, Y, epochs):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - Y)**2)
            losses.append(loss)
            self.backward(X, Y)

        return losses