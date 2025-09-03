import numpy as np
from PIL import Image
from pathlib import Path
import os
import matplotlib.colors as mcolors
from pathlib import Path
import csv
from color_conversion import HEXtoLAB
from clustering import cluster_lab_colors



def load_and_convert_data(file_name):
    data = []
    with open(file_name, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:
                image_name = parts[0]
                hex_colors = parts[1:]
                lab_colors = [HEXtoLAB(c.strip(',')) for c in hex_colors]
                lab_colors = np.asarray(lab_colors).flatten()
                data.append((image_name, lab_colors))
    
    data.reverse()
    return data


def create_data_CSV(csv_file_name, raw_data_file, images_folder):
    data = load_and_convert_data(raw_data_file)
    rows = []
    folder = Path(images_folder)

    
    for image_name, colors in data:
        image_path = folder / image_name
        if image_path.exists():
            image_tensor = cluster_lab_colors(image_path).flatten()
            row = [image_name, * colors.tolist(), * image_tensor.tolist()]
            rows.append(row)




    with open(csv_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["file_name"]
        for i in range(1, 6):
            header += [f"l(real)_{i}", f"a(real)_{i}", f"b(real)_{i}"]
        for i in range(1, 6):
            header += [f"l_{i}", f"a_{i}", f"b_{i}"]
        writer.writerow(header)
        writer.writerows(rows)
            


    
        



            

