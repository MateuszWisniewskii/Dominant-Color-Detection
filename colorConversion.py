import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import matplotlib.colors as mcolors
from pathlib import Path


LAB_L_MAX = 100.0
LAB_AB_SHIFT = 128
LAB_AB_MAX = 255.0

def normalizeLabTensor(lab_tensor):

    lab_tensor = lab_tensor.copy()

    #change to 2 dimension tensor if was 1 dimension
    if lab_tensor.ndim == 1:
        lab_tensor[0] = lab_tensor[0] / LAB_L_MAX
        lab_tensor[1] = (lab_tensor[1] + LAB_AB_SHIFT) / LAB_AB_MAX
        lab_tensor[2] = (lab_tensor[2] + LAB_AB_SHIFT) / LAB_AB_MAX
        return lab_tensor

    lab_tensor[:, 0] = lab_tensor[:, 0] / LAB_L_MAX
    lab_tensor[:, 1] = (lab_tensor[:, 1] + LAB_AB_SHIFT) / LAB_AB_MAX
    lab_tensor[:, 2] = (lab_tensor[:, 2] + LAB_AB_SHIFT) / LAB_AB_MAX

    return lab_tensor


def RGBtoLAB(image_path):

    #open -> convert to rgb -> resize to 64x64 return object PIL.Image.Image
    image = Image.open(image_path).convert('RGB')
    #convert PIL object to NumPy array [Height, Width, RGB channels] and normalize pixels from 0-255 to 0-1
    image_np = np.asarray(image) / 255
    #convert from RGB to Lab and change from [H, W, C] to [C, H, W]
    lab_image = rgb2lab(image_np).transpose(2, 0, 1)

    lab_image = normalizeLabTensor(lab_image)

    #change array from [Height, Width, Channels] to [Channels, Height, Width]
    return lab_image



def HEXtoLAB(hex_color):
    #mcolors.to_rgb -> creates tuple and changes hex value to rgb
    #np.array -> convert tuple to numPy array
    #reshape -> creates image 1x1 pixel and 3 channels rgb
    rgb = np.array(mcolors.to_rgb(hex_color)).reshape(1,1,3)
    #change from rgb to Lab
    lab = rgb2lab(rgb).flatten()

    lab = normalizeLabTensor(lab)

    #return flat vertor [3,]
    return lab.flatten()