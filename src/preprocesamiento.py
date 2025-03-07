from PIL import ImageEnhance
from skimage import restoration, filters, feature
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st 



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

######### Binarización de Otsu #########
def binarizar_otsu(imagen):
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen.convert('L'))
    elif len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    _, imagen_binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #_ es el umbral pero no lo necesitamos
    
    return Image.fromarray(imagen_binarizada)

# Función para binarización manual
def binarizar_manual(imagen, umbral):
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen.convert('L'))
    elif len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, imagen_binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return Image.fromarray(imagen_binarizada)

#Método para segmentación de imagen por umbral
def segmentar_umbral(imagen, umbral):
    return imagen > umbral

def segmentar_bordes(imagen, sigma=1.0):
    return feature.canny(imagen, sigma=sigma)
