import torch
from neuralNetwork import ConvolutionalNeuralNetwork
from prepareData import RGBtoLAB
from pathlib import Path
import numpy as np
import argparse
from skimage.color import lab2rgb
import matplotlib.colors as mcolors
from PIL import Image


#function for displaying image containing square in given color
def showColor(hex_color):
    #convert hex color to rgb (0..1 values returned) multiply that by 255 to get full range of rgb
    rgb = tuple(int(255 * c) for c in mcolors.to_rgb(hex_color))
    #create image 100x100 colored in calculated rgb
    img = Image.new("RGB", (100, 100), rgb)
    #display image
    img.show()


#function that takes given image, prediscts color in Lab from trained model and converts output to hex value
def predict(image_path):
    #load image and convert from RGB -> Lab, normalize, change dimention to (1, 3, 64, 64)->(quantity, channels, size, size)
    image_tensor = RGBtoLAB(image_path)
    image_tensor = torch.tensor(image_tensor).float().unsqueeze(0)

    #load trained model from file
    model = ConvolutionalNeuralNetwork()
    model.load_state_dict(torch.load("model.pt"))

    model.eval()

    #turn off gradients for faster computing (no need for gradients in prediction bcs we are not updaing weights)
    with torch.no_grad():
        output = model(image_tensor).squeeze().numpy()


    #denormalization of output
    output[0] = output[0] * 255
    output[1] = output[1] * 255 - 128
    output[2] = output[2] * 255 - 128

    #Lab->RGB conversion
    rgb = lab2rgb(output)

    #RGB->HEX converion
    hex_color = mcolors.to_hex(rgb)

    showColor(hex_color)

    print("Predicted LAB:", hex_color)





if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image.")

    args = parser.parse_args()

    predict(args.image)

