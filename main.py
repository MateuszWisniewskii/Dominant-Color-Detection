import numpy as np
from PIL import Image
import imageLoader as il
from dominantColorAlgorithm import findDominantColor
import neuralNetwork as nn
import matplotlib.pyplot as plt


loader = il.ImageLoader()

arr = loader.open("Data/")

normalizedInput = loader.normalizeImage()

img = Image.open("Data/image2.jpeg")

color = findDominantColor(img)

print("Real values:",color)

#rint(normalizedInput)


net = nn.NeuralNetwork(learningRate=0.01)

losses = net.train(normalizedInput, color, 100)

prediction = net.forward(normalizedInput)

print("Prediction (output A3):", prediction)

plt.plot(range(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Error (Loss)")
plt.title("Neural Network training loss")
plt.grid(True)
plt.show()
