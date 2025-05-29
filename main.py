import numpy as np
from PIL import Image
import imageLoader as il
from dominantColorAlgorithm import findDominantColor
import neuralNetwork as nn
import matplotlib.pyplot as plt
import argparse


def Predict(image):
    net = nn.NeuralNetwork(learningRate=0.01)
    imageLoader = il.ImageLoader()
    net.loadModel()
    normalizedInput = imageLoader(image)
    prediction = net.forward(normalizedInput)
    print("Prediction: ", prediction)


def Train(pathToDataset):
    net = nn.NeuralNetwork(learningRate=0.01)
    imageLoader = il.ImageLoader()
    lossesArray = []

    imageLoader.open(pathToDataset)
    normalizedImages = imageLoader.normalizeImage()

    for i in range(len(normalizedImages)):
        Y = findDominantColor(imageLoader.images[i])
        X = normalizedImages[i].reshape(1,-1)
        epochs = 10
        losses = net.train(X, Y, epochs)
        lossesArray.append(losses)

    for i, losses in enumerate(lossesArray):
        plt.plot(range(len(losses)), losses, label=f'Image {i+1}')
    plt.xlabel("Epochs")
    plt.ylabel("Error (Loss)")
    plt.title("Neural Network training loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or predict using a simple neural network.")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help='train or predict')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--dataset', type=str, help='Folder with data set')

    args = parser.parse_args()

    if args.mode == 'predict':
        if not args.image:
            parser.error("Use --image [path to image] for prediction")
        Predict(args.image)
    if args.mode == 'train':
        if not args.dataset:
            parser.error("Use --dataset [folder with data set] for training")
        Train(args.dataset)

    






    
