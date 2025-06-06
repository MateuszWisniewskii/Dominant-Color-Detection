import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # First convolutional layer: input has 3 LAB channels (L, a, b), outputs 16 feature maps
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # Second convolutional layer: input 16 feature maps, outputs 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # First fully connected layer: flatten output from conv2 and reduce to 64 features
        # Input size assumes input LAB image size of 64x64, after 2x max pooling: 64 -> 32 -> 16
        self.fc1 = nn.Linear(32 * 16 * 16, 64)

        # Output layer: 3 output neurons, e.g. for predicting 3 dominant color components
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Apply first convolution + ReLU activation (LAB input)
        x = F.relu(self.conv1(x))

        # Downsample with max pooling (reduce spatial size by 2)
        x = F.max_pool2d(x, 2)

        # Apply second convolution + ReLU
        x = F.relu(self.conv2(x))

        # Second max pooling
        x = F.max_pool2d(x, 2)

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer + ReLU
        x = F.relu(self.fc1(x))

        # Output layer + Sigmoid activation for normalized output (0–1)
        x = F.sigmoid(self.fc2(x))  # You can replace this with torch.sigmoid(x) if using PyTorch > 1.7

        return x
