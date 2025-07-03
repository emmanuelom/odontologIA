from PIL import ImageEnhance
from skimage import restoration, filters, feature
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st 



def logo_segun_tema():
    url_logo_claro = "https://www.lasallebajio.edu.mx/comunidad/images/imagotipos/lasallebajio_NEGRO.png"
    url_logo_oscuro = "https://www.lasallebajio.edu.mx/comunidad/images/imagotipos/lasallebajio_BLANCO.png"
    opcion = st.sidebar.radio("Selecciona el tema del logo", ["Claro", "Oscuro"])
    logo_url = url_logo_claro if opcion == "Claro" else url_logo_oscuro
    st.markdown(
        f'<div style="display:flex;justify-content:center;"><img src="{logo_url}" width="300"></div>',
        unsafe_allow_html=True
    )

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

# Función para binarización por rango de umbrales
def binarizar_rango(imagen, umbral_min, umbral_max):
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen.convert('L'))
    elif len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    imagen_normalizada = imagen / 255.0  # Escalar a 0-1
    imagen_binarizada = (imagen_normalizada >= umbral_min) & (imagen_normalizada <= umbral_max)
    imagen_binarizada = (imagen_binarizada * 255).astype(np.uint8)  # Convertir a 0-255
    
    return Image.fromarray(imagen_binarizada)

#Método para segmentación de imagen por umbral
def segmentar_umbral(imagen, umbral):
    return imagen > umbral

def segmentar_bordes(imagen, sigma=1.0):
    return feature.canny(imagen, sigma=sigma)

######### Operadores morfológicos #########
def erosionar(imagen, kernel_size=(3, 3), iterations=1):
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen)
    kernel = np.ones(kernel_size, np.uint8)
    imagen_erosionada = cv2.erode(imagen, kernel, iterations=iterations)
    return Image.fromarray(imagen_erosionada)

def dilatar(imagen, kernel_size=(3, 3), iterations=1):
    if isinstance(imagen, Image.Image):
        imagen = np.array(imagen)
    kernel = np.ones(kernel_size, np.uint8)
    imagen_dilatada = cv2.dilate(imagen, kernel, iterations=iterations)
    return Image.fromarray(imagen_dilatada)