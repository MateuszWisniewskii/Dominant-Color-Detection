import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import matplotlib.colors as mcolors
from pathlib import Path

LAB_L_MAX = 100.0
LAB_AB_SHIFT = 128
LAB_AB_MAX = 255.0

# Jeśli dostaje jeden kolor [L, a, b] -> skaluje każdy kanał do [0,1]
# Jeśli dostaje listę kolorów [[L,a,b], [L,a,b], ...] -> robi to samo dla całej tablicy
def normalize_lab_tensor(lab_tensor):
    lab_tensor = lab_tensor.copy() # robimy kopię, żeby nie modyfikować oryginału przekazanego jako argument

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

# Przykład:

# lab_tensor = np.array([
#     [50,   0,   0],
#     [20, -30,  10],
#     [75,   5,  20],
#     [62,  40, -25],
#     [10,   0,  -2]
# ])
# : -> oznacza „wszystkie elementy w tym wymiarze”.
# lab_tensor[:, 0]
# pierwszy indeks : -> bierzemy wszystkie wiersze (czyli wszystkie kolory)
# drugi indeks 0 -> ale tylko kolumnę 0 (czyli L).
# wynik: [50, 20, 75, 62, 10]
# lab_tensor[:, 1]
# wszystkie wiersze, ale tylko kolumna 1 (czyli a).
# wynik: [0, -30, 5, 40, 0]
# lab_tensor[:, 2]
# wszystkie wiersze, ale tylko kolumna 2 (czyli b).
# wynik: [0, 10, 20, -25, -2]