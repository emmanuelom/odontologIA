from PIL import Image
import io

def cargar_imagen(imagen_subida):
    imagen = Image.open(imagen_subida)
    return imagen

def guardar_imagen(imagen, ruta):
    imagen.save(ruta)