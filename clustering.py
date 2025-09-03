from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
from normalize_lab_tensor import normalize_lab_tensor

def cluster_lab_colors(image_path, colors=5):
    # Wczytanie obrazu
    image = Image.open(image_path).convert("RGB") # "./Data/PhotosColorPicker/000000010432.jpg"
    image_np = np.asarray(image)/255.0

    # Konwersja z RGB do LAB
    lab_image = rgb2lab(image_np)
    pixels = lab_image.reshape(-1,3)

    # print(lab_image.shape)
    # pixel = image_np[1, 0]
    # print(pixel)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)

    centroids = kmeans.cluster_centers_

    # i = 0
    # while i < len(centroids):
    #     print(centroids[i])
    #     i+=1

    centroids = normalize_lab_tensor(centroids)

    # print("=============================================")

    # i = 0
    # while i < len(centroids):
    #     print(centroids[i])
    #     i+=1

    return centroids