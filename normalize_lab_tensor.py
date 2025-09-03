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