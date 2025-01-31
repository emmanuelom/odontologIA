from PIL import ImageEnhance

######### Seleccionar region #########
def seleccionar_region(imagen, left, top, right, bottom):
    region = imagen.crop((left, top, right, bottom))
    return region

######### Mejorar contraste #########
def mejorar_contraste(imagen, factor):
    contraste = ImageEnhance.Contrast(imagen)
    imagen_mejorada = contraste.enhance(factor)
    return imagen_mejorada