from PIL import Image
import numpy as np


def findDominantColor(image):
    img = image.copy().convert("RGB")
    img = img.resize((64, 64))
    pixels = np.array(img)
    pixels = pixels.reshape(-1, 3)

    mean_color = np.mean(pixels, axis=0) / 255  
    mean_color = np.round(mean_color, 1)        

    return np.array(mean_color)