import numpy as np


class NeuralNetwork:
    def __init__(self, learningRate):

        self.path="modelWeights.npz"

        self.learningRate = learningRate

        #He weights initialization for ReLU
        self.weights1 = np.random.randn(12288, 32) * np.sqrt(2. / 12288)
        self.bias1 = np.zeros((1, 32))

        #He weights initialization for ReLU
        self.weights2 = np.random.randn(32, 16) * np.sqrt(2. / 32)
        self.bias2 = np.zeros((1, 16))

        #Xavier initialization for Sigmoid
        self.weights3 = np.random.randn(16, 3) * np.sqrt(1. / 16)
        self.bias3 = np.zeros((1, 3))


        #More neurons = better results but for this simple task like finding dominant color there is no need for more neurons

        # #He weights initialization for ReLU
        # self.weights1 = np.random.randn(12288, 64) * np.sqrt(2. / 12288)
        # self.bias1 = np.zeros((1, 64))

        # #He weights initialization for ReLU
        # self.weights2 = np.random.randn(64, 32) * np.sqrt(2. / 64)
        # self.bias2 = np.zeros((1, 32))

        # #Xavier initialization for Sigmoid
        # self.weights3 = np.random.randn(32, 3) * np.sqrt(1. / 32)
        # self.bias3 = np.zeros((1, 3))


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

        # Batch size (number of samples)
        batch = X.shape[0]

        # Output layer error
        dA3 = self.A3 - Y
        dZ3 = dA3 * self.sigmoid_derivative(self.Z3)  # Derivative through sigmoid

        # Hidden layer 2 error
        dA2 = dZ3 @ self.weights3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)     # Derivative through ReLU

        # Hidden layer 1 error
        dA1 = dZ2 @ self.weights2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)     # Derivative through ReLU

        # Gradients for weights and biases
        dW3 = self.A2.T @ dZ3 / batch                 # Gradient of weights between layer 2 and output
        db3 = np.sum(dZ3, axis=0, keepdims=True) / batch  # Gradient of biases for output layer

        dW2 = self.A1.T @ dZ2 / batch                 # Gradient of weights between layer 1 and 2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch  # Gradient of biases for layer 2

        dW1 = X.T @ dZ1 / batch                       # Gradient of weights between input and layer 1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch  # Gradient of biases for layer 1

        # Update weights and biases
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
    

    def saveModel(self):
        np.savez(self.path, weights1=self.weights1,
             bias1=self.bias1,
             weights2=self.weights2,
             bias2=self.bias2,
             weights3=self.weights3,
             bias3=self.bias3)
        return{f"Model saved to {self.path}"}
    

    def loadModel(self):
        data = np.load(self.path)
        self.weights1 = data['weights1']
        self.bias1 = data['bias1']
        self.weights2 = data['weights2']
        self.bias2 = data['bias2']
        self.weights3 = data['weights3']
        self.bias3 = data['bias3']
        print(f"Model loaded from {self.path}")