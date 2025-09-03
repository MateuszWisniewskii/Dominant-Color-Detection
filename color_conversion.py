import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import matplotlib.colors as mcolors
from pathlib import Path
from normalize_lab_tensor import normalize_lab_tensor

def RGBtoLAB(image_path):

    #open -> convert to rgb -> resize to 64x64 return object PIL.Image.Image
    image = Image.open(image_path).convert('RGB')
    #convert PIL object to NumPy array [Height, Width, RGB channels] and normalize pixels from 0-255 to 0-1
    image_np = np.asarray(image) / 255
    #convert from RGB to Lab and change from [H, W, C] to [C, H, W]
    lab_image = rgb2lab(image_np).transpose(2, 0, 1)

    lab_image = normalize_lab_tensor(lab_image)

    #change array from [Height, Width, Channels] to [Channels, Height, Width]
    return lab_image



def HEXtoLAB(hex_color):
    #mcolors.to_rgb -> creates tuple and changes hex value to rgb
    #np.array -> convert tuple to numPy array
    #reshape -> creates image 1x1 pixel and 3 channels rgb
    rgb = np.array(mcolors.to_rgb(hex_color)).reshape(1,1,3)
    #change from rgb to Lab
    lab = rgb2lab(rgb).flatten()

    lab = normalize_lab_tensor(lab)

    #return flat vertor [3,]
    return lab.flatten()