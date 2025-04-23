import numpy as np
from PIL import Image
from pathlib import Path


#This class iterates through directory and opens images, resizes them to 64x64 and normalizes.
# images -> array of images
# normalized -> array of normalizes images
class ImageLoader:
    def __init__(self, deafultSize=(64,64)):
        self.deafultSize = deafultSize
        self.images = []
        self.normalized = []
        

    def open(self, folderPath):
        folder = Path(folderPath)
        for filePath in folder.iterdir():
            if filePath.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    img = Image.open(filePath).resize(self.deafultSize)
                    self.images.append(img)
                    print(f"opened: {filePath}")
                except IOError as error:
                    print(error)
                    pass
    
    #Input normalization from RBG -> 0-255 to 0-1 values  
    def normalizeImage(self):
        self.normalized = []
        for image in self.images:
            arr = np.array(image) / 255.0
            self.normalized.append(arr)

        