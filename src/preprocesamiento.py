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

######### Umbral óptimo con histograma y CDF #########
def encontrar_umbral_optimo(imagen):
    """
    Encuentra el umbral óptimo usando Otsu y genera histograma y CDF
    Retorna: umbral_optimo, imagen_binarizada, figura_histograma
    """
    if isinstance(imagen, Image.Image):
        imagen_array = np.array(imagen.convert('L'))
    elif len(imagen.shape) == 3:
        imagen_array = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_array = imagen

    # Calcular umbral de Otsu AQUI SE APLICA EL UMBRAL OPTIMO, el calculo se realizó con ayuda de IA
    umbral_optimo, imagen_binarizada = cv2.threshold(imagen_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Crear histograma y CDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Histograma
    hist, bins = np.histogram(imagen_array.flatten(), bins=256, range=[0, 256])
    ax1.bar(bins[:-1], hist, width=1, color='gray', alpha=0.7)
    ax1.axvline(x=umbral_optimo, color='red', linestyle='--', linewidth=2, label=f'Umbral óptimo: {int(umbral_optimo)}')
    ax1.set_title('Histograma')
    ax1.set_xlabel('Intensidad de píxel')
    ax1.set_ylabel('Frecuencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CDF (Función de Distribución Acumulativa)
    cdf = np.cumsum(hist)
    cdf_normalized = cdf * hist.max() / cdf.max()
    ax2.plot(bins[:-1], cdf_normalized, color='blue', linewidth=2)
    ax2.axvline(x=umbral_optimo, color='red', linestyle='--', linewidth=2, label=f'Umbral óptimo: {int(umbral_optimo)}')
    ax2.set_title('CDF Normalizada')
    ax2.set_xlabel('Intensidad de píxel')
    ax2.set_ylabel('CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return int(umbral_optimo), Image.fromarray(imagen_binarizada), fig

