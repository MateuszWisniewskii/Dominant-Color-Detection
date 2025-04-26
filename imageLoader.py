import numpy as np
from PIL import Image
from pathlib import Path


#This class iterates through directory and opens images, resizes them to 64x64 and normalizes.
# images -> array of images
# normalized -> array of normalizes images
class ImageLoader:
    def __init__(self, defaultSize=(64,64)):
        self.defaultSize = defaultSize
        self.images = []
        

    def open(self, folderPath):
        folder = Path(folderPath)
        for filePath in folder.iterdir():
            if filePath.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    img = Image.open(filePath).resize(self.defaultSize)
                    self.images.append(img)
                    print(f"opened: {filePath}")
                except IOError as error:
                    print(error)
                    pass
        
        return self.images

    
    #Input normalization from RBG -> 0-255 to 0-1 values  
    def normalizeImage(self):
        normalized = []
        for image in self.images:
            img = np.array(image, dtype=np.float32) / 255.0
            img = img.flatten()
            img = np.round(img, 1)
            normalized.append(img)
        return np.array(normalized)

        