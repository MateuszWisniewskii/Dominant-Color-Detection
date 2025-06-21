import torch
from neuralNetwork import ConvolutionalNeuralNetwork
from colorConversion import RGBtoLAB
from pathlib import Path
import numpy as np
import argparse
from skimage.color import lab2rgb
import matplotlib.colors as mcolors
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt


def denormalizeLabColor(color):
    return np.array([
        color[0] * 100,
        color[1] * 255 - 128,
        color[2] * 255 - 128])


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

    target_height, target_width = 480, 640
    #load image and convert from RGB -> Lab, normalize, change dimention to (1, 3, 64, 64)->(quantity, channels, size, size)
    image_tensor = RGBtoLAB(image_path)
    image_tensor = torch.tensor(image_tensor).float()


    c, h, w = image_tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w

    padding = (0, pad_w, 0 , pad_h)

    padded_image = F.pad(torch.tensor(image_tensor), padding, mode='constant', value=0)
    #load trained model from file
    model = ConvolutionalNeuralNetwork()
    model.load_state_dict(torch.load("model.pt"))

    model.eval()

    #turn off gradients for faster computing (no need for gradients in prediction bcs we are not updaing weights)
    with torch.no_grad():
        output = model(padded_image.unsqueeze(0)).squeeze().numpy()


    predicted_colors = output.reshape(5,3)

    denormalized_predicted_colors = np.asarray([denormalizeLabColor(c) for c in predicted_colors])

    #Lab->RGB conversion
    rgb_predicted_colors = np.asarray([lab2rgb(c) for c in denormalized_predicted_colors])

    #RGB->HEX converion
    hex_predicted_colors = np.asarray([mcolors.to_hex(c) for c in rgb_predicted_colors])

    print("Predicted LAB:", hex_predicted_colors)

    fig, ax = plt.subplots(1, 5, figsize=(12, 3))
    for i, (color, hex_val) in enumerate(zip(rgb_predicted_colors, hex_predicted_colors)):
        ax[i].imshow(np.ones((100, 100, 3)) * color.reshape(1, 1, 3))
        ax[i].axis('off')
        ax[i].set_title(hex_val, fontsize=10)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image.")

    args = parser.parse_args()

    predict(args.image)

