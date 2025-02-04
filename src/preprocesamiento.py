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
    contraste = ImageEnhance.Contrast(imagen)
    imagen_mejorada = contraste.enhance(factor)
    return imagen_mejorada

######### Mejorar contraste con CLAHE #########
def mejorar_contraste_clahe(imagen, clahe_clip=2.0, clahe_grid=(8, 8)):
    imagen_np = np.array(imagen.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    imagen_clahe = clahe.apply(imagen_np)
    return Image.fromarray(imagen_clahe)

######### Mejorar enfoque usando ROLLING_BALL #########
def mejorar_enfoque(imagen, focus_radius=50):
    imagen_np = np.array(imagen.convert('L'))
    background = restoration.rolling_ball(imagen_np, radius=focus_radius)
    imagen_enfocada = imagen_np - background
    return Image.fromarray(imagen_enfocada)


