import streamlit as st
from skimage import io
import numpy as np

def cargar_imagen(imagen_subida):
    imagen = io.imread(imagen_subida)
    return imagen

def guardar_imagen(imagen, ruta):
    io.imsave(ruta, imagen)