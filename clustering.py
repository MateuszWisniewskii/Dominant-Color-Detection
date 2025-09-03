from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
from color_conversion import normalizeLabTensor

# Wczytanie obrazu
image = Image.open("./Data/PhotosColorPicker/000000010432.jpg").convert("RGB")
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

i = 0
while i < len(centroids):
    print(centroids[i])
    i+=1

centroids = normalizeLabTensor(centroids)

print("=============================================")

i = 0
while i < len(centroids):
    print(centroids[i])
    i+=1