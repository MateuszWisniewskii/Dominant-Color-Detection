import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        #first convolutional layer: input has 3 LAB channels (L, a, b), outputs 16 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        #second convolutional layer: input 16 feature maps, outputs 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        #fully connected layer (32 channels from conv2 -> 64 hidden units
        self.fc1 = nn.Linear(64, 64)

        #output layer: 3 output neurons, e.g. for predicting 3 dominant color components
        self.fc2 = nn.Linear(64, 15)

    def forward(self, x):
        #apply first convolution + ReLU activation (LAB input)
        x = F.relu(self.conv1(x))

        #downsample with max pooling (reduce spatial size by 2)
        x = F.max_pool2d(x, 2)

        #apply second convolution + ReLU
        x = F.relu(self.conv2(x))

        #second max pooling
        x = F.max_pool2d(x, 2)

        #global average pooling across spatial dimensions -> (B, 32, 1, 1)
        x = self.gap(x)

        #flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        #fully connected layer + ReLU
        x = F.relu(self.fc1(x))

        #output layer + Sigmoid activation for normalized output (0–1)
        x = F.sigmoid(self.fc2(x))  # You can replace this with torch.sigmoid(x) if using PyTorch > 1.7

        return x
