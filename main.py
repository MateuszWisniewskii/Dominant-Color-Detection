import numpy as np
from PIL import Image
import imageLoader as il
from dominantColorAlgorithm import findDominantColor
import neuralNetwork as nn


loader = il.ImageLoader()

arr = loader.open("Data/")

normalizedInput = loader.normalizeImage()

img = Image.open("Data/image1.png")

color = findDominantColor(img)

print(color)

print(normalizedInput)


net = nn.NeuralNetwork(learningRate=0.01)

net.train(normalizedInput, color, 100)

prediction = net.forward(normalizedInput)

print("Prediction (output A3):", prediction)
