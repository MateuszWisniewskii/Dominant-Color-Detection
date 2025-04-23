import numpy as np
from PIL import Image
from pathlib import Path



class ImageLoader:
    images = list()

    def __init__(self, deafultSize=(64,64)):
        self.deafultSize = deafultSize
        

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
        return 0