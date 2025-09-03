import numpy as np
from PIL import Image
from pathlib import Path
import os
import matplotlib.colors as mcolors
from pathlib import Path
import csv
from color_conversion import HEXtoLAB
from clustering import cluster_lab_colors


# Funkcja wczytuje dane z pliku tekstowego i konwertuje kolory HEX na LAB
def load_and_convert_data(file_name):
    data = [] # Lista danych o obrazach i kolorach
    with open(file_name, "r") as file:
        for line in file:
            parts = line.strip().split() # Dzielimy linię na elementy (białe znaki jako separator)
            if len(parts) == 6: # Sprawdzamy, czy linia zawiera 6 elementów (1 nazwa pliku + 5 kolorów HEX)
                image_name = parts[0]
                hex_colors = parts[1:] # Lista kolorów HEX
                lab_colors = [HEXtoLAB(c.strip(',')) for c in hex_colors]  # Konwertujemy każdy kolor HEX na LAB i zapisujemy jako tablicę
                lab_colors = np.asarray(lab_colors).flatten() # Spłaszczamy do jednego wymiaru
                data.append((image_name, lab_colors))
    
    data.reverse()
    return data

# Funkcja tworzy plik CSV z danymi o kolorach obrazów
def create_data_CSV(csv_file_name, raw_data_file, images_folder):
    # Wczytanie surowych danych z pliku .txt
    data = load_and_convert_data(raw_data_file)
    rows = []
    folder = Path(images_folder)

    # Klasteryzacja -> wyciągnięcie dominujących kolorów dla każdego z obrazków
    for image_name, colors in data:
        image_path = folder / image_name
        if image_path.exists():
            image_tensor = cluster_lab_colors(image_path).flatten() # wyciąga dominujące kolory z obrazu (LAB, znormalizowane)
            row = [image_name, * colors.tolist(), * image_tensor.tolist()]
            rows.append(row)

    # Tworzenie pliku .csv
    with open(csv_file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["file_name"]
        for i in range(1, 6):
            header += [f"l(real)_{i}", f"a(real)_{i}", f"b(real)_{i}"] # kolory z pliku
        for i in range(1, 6):
            header += [f"l_{i}", f"a_{i}", f"b_{i}"] # kolory wyekstrahowane z obrazu
        writer.writerow(header)
        writer.writerows(rows)
            


    
        



            

