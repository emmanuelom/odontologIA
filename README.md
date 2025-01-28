# odontologIA

## Objetivos

Desarrollar una interfaz amigable para la mejora de imágenes dentales (de rayos X). 

## Paquetes

El software se desarrolla usando Python >= 3.10 usando las librerias:

* Streamlit 
* OpenCV
* Skimage
* Numpy

## Dataset

La base de datos se encuentra en la carpeta dataset/

## Organización del proyecto

* En la carpeta src/ incluir los códigos fuente (.py)
    * app.py : Interfaz de Streamlit, carga, vizualización de la imagen y llamada a los modulos de procesamiento de imágenes
    * preprocesamiento.py : Métodos para preprocesar la imagen
    * io_img.py: Métodos para la lectura y escritura de imagenes

* En la ruta raiz, agregar requirements.txt

## Notas:

* Crear un ambiente virtual (local) con conda o venv
* Agregar .gitignore para no subir paquetes y/o archivos cache
* Manejar rutas relativas
* Recuerda hacer pull antes de push. Manten actualizado tu repo ante cambios remotos.