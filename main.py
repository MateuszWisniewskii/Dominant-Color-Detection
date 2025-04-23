import numpy as np
#from PIL import Image
from imageLoader import ImageLoader

# short for separator
def sep():
    print("=============================================================")

images_folder = "images_from_teams/"

# magic number by now, 'couse images in /images_fro_teams are in many different sizes, some of tehem are jutr taller than wider
image_X = 300
image_Y = 300
image_size = (image_X, image_Y) 

input_layer_nerurons_amount = image_X * image_Y
output_layer_nerurons_amount = 3 # Only three because we are going to ingnore alpha factor

neurons_in_layer = [input_layer_nerurons_amount, output_layer_nerurons_amount]


loader = ImageLoader(image_size)

loader.open(images_folder)



sep()
