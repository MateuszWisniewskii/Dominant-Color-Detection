from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
from normalize_lab_tensor import normalize_lab_tensor

def cluster_lab_colors(image_path, colors=5):
    image = Image.open(image_path).convert("RGB") # Wczytanie obrazu i konwersja do przestrzeni RGB
    image_np = np.asarray(image)/255.0 # normalizacja pikseli (0–1)

    # Konwersja z RGB do LAB L -> jasność (0–100)
    # a -> oś zielony <-> czerwony (około -128 – +127)
    # b -> oś niebieski <-> żółty (około -128 – +127)
    lab_image = rgb2lab(image_np) 
    pixels = lab_image.reshape(-1,3) # spłaszczenie wszystkich pikseli w jedną listę wektorów
                                     # -1 oznacza, że "Python sam policzy ile wierszy ma być"
                                     # 3 -> bo każdy piksel ma 3 wartości: [L, a, b]

    # n_clusters -> Liczba klastrów, które algorytm ma znaleźć. 5 bo tyle dominujących kolorów chcemy znaleźć
    # random_state -> seed generatora losowego
    # n_init=10 -> liczba uruchomień algorytmu                                
    kmeans = KMeans(n_clusters=colors, random_state=42, n_init=10)
    kmeans.fit(pixels) # 1. Losowo wybiera n_clusters punktów startowych (centroidów) z przestrzeni pikseli (np. 5 punktów [L, a, b])
                       # 2. Dla każdego piksela oblicza, do którego centroidu jest najbliżej 
                       # 3. Grupuje piksele w klastry (każdy piksel należy do tego centroidu, który jest mu najbliższy)
                       # 4. Przesuwa centroid każdego klastra do średniej wszystkich pikseli, które do niego należą
                       # 5. Powtarza kroki 2–4 aż do stabilizacji (czyli centroidy już prawie się nie zmieniają)

    centroids = kmeans.cluster_centers_ # pzypisanie centroidów, czyli dominujących kolorów obrazu
    centroids = normalize_lab_tensor(centroids) # normalizacja wartości Lab do zakresu od 0 do 1
                                                # (L jest w zakresie 0–100 natomiast a oraz b są mniej więcej w zakresie -128 – +127)
    return centroids