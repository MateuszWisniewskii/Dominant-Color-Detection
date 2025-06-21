import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
from colorConversion import normalizeLabTensor

def clusterLabColors(image_path, colors=5):
    image = Image.open(image_path).convert('RGB')
    image_np = np.asarray(image) / 255.0

    lab_image = rgb2lab(image_np)
    pixels = lab_image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    centroids = kmeans.cluster_centers_

    centroids = normalizeLabTensor(centroids)
    
    return centroids
