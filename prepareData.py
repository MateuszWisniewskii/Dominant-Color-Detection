import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab
import os
import torch
import matplotlib.colors as mcolors
from pathlib import Path
import csv

def normalizeLabTensor(lab_tensor):

    lab_tensor = lab_tensor.copy()
    lab_tensor[0] = lab_tensor[0] / 255
    lab_tensor[1] = (lab_tensor[1] + 128) / 255.0
    lab_tensor[2] = (lab_tensor[2] + 128) / 255.0

    return lab_tensor


#Function that opens an image, resize it and converts it from RGB to Lab
def RGBtoLAB(image_path, default_size=(64,64)):

    #open -> convert to rgb -> resize to 64x64 return object PIL.Image.Image
    image = Image.open(image_path).convert('RGB').resize(default_size)
    #convert PIL object to NumPy array [64,64,3] -> [Height, Width, RGB channels] and normalize pixels from 0-255 to 0-1
    image_np = np.asarray(image)
    #convert from RGB to Lab
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
    

def loadAndConvertData(file_name):
    data = []
    with open(file_name, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                image_name, hex_color = parts
                data.append((image_name, HEXtoLAB(hex_color)))
    
    data.reverse()
    return data


def createDataCSV(csv_file_name):
    data = loadAndConvertData("/Users/patrykpandel/Desktop/BIAI/Data/GKew_2025_03_20-11_03_35.txt")
    images = []
    rows = []
    folder = Path("/Users/patrykpandel/Desktop/BIAI/Data/PhotosColorPicker")

    
    for image_name, Lab_label in data:
        image_path = folder / image_name
        if image_path.exists():
            image_tensor = RGBtoLAB(image_path).flatten()
            row = [image_name] + Lab_label.tolist() + image_tensor.tolist()
            rows.append(row)




    with open(csv_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["file_name", "L", "a", "b"] + [f"pixel_{i}" for i in range(12288)]
        writer.writerow(header)
        writer.writerows(rows)
            
    

    
        



            


