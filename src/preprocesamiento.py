from PIL import ImageEnhance
from skimage import restoration
import numpy as np
import cv2
from PIL import Image


######### Seleccionar region #########
def seleccionar_region(imagen, left, top, right, bottom):
    region = imagen.crop((left, top, right, bottom))
    return region

######### Mejorar contraste #########
def mejorar_contraste(imagen, factor):
    if isinstance(imagen, np.ndarray):
        imagen = Image.fromarray(imagen)
    if imagen.mode != 'L':
        imagen = imagen.convert('L')
    
    contraste = ImageEnhance.Contrast(imagen)
    return np.array(contraste.enhance(factor))

######### Mejorar contraste con CLAHE #########
def mejorar_contraste_clahe(imagen, clahe_clip=2.0, clahe_grid=(8, 8)):
    if isinstance(imagen, Image.Image):  
        imagen = np.array(imagen.convert('L'))  

    imagen = np.asarray(imagen, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    imagen_clahe = clahe.apply(imagen)

    return Image.fromarray(imagen_clahe)




