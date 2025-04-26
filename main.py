import numpy as np
from PIL import Image
from imageLoader import ImageLoader
from dominantColorAlgorithm import findDominantColor


loader = ImageLoader()

loader.open("Data/")


img = Image.open("Data/image1.png")

color = findDominantColor(img)

print(color)