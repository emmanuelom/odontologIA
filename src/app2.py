import streamlit as st
from streamlit_cropper import st_cropper
from io_img import cargar_imagen, guardar_imagen
from preprocesamiento import mejorar_contraste, seleccionar_region, mejorar_contraste_clahe, binarizar_otsu, binarizar_manual, segmentar_umbral, segmentar_bordes, erosionar, dilatar, binarizar_rango
from PIL import Image
import numpy as np
from skimage import color
from skimage.transform import resize, rescale, downscale_local_mean

# Estilos
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        padding: 10px 50px;
        margin-bottom: 10px;
        font-size: 16px;
        color: white;
        background-color: #43aa8b;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
        color: white;
        border: white;
    }
    .stTitle {
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
    
    .center {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Inicio
# Imagen institucional
st.markdown('<div class="center"><img src="https://www.lasallebajio.edu.mx/comunidad/images/imagotipos/Elementos%20Gráficos/Salle%20Bajío%2011.png" width="300"></div>', unsafe_allow_html=True)
st.markdown("""<h1 class='stTitle'>Bienvenido a odontolog<span style='color: #4CAF50;'>IA</span></h1>""", unsafe_allow_html=True)
st.markdown("<p class='stMarkdown'>Carga una imagen para comenzar</p>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <h1 style='font-size: 50px;'>
        odontolog<span style='color: #4CAF50;'>IA</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# Inicializar claves en st.session_state si no existen
if 'region_seleccionada' not in st.session_state:
    st.session_state.region_seleccionada = None
if 'region_binarizada' not in st.session_state:
    st.session_state.region_binarizada = None
if 'region_bordes' not in st.session_state:
    st.session_state.region_bordes = None

# SEC-1: Cargar una imagen
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="file_uploader")
if imagen_subida is not None:
    imagen = cargar_imagen(imagen_subida)
    
    # Conversión img skimage array a img PIL
    if isinstance(imagen, np.ndarray):
        imagen = Image.fromarray(imagen)
    imagen = color.rgb2gray(np.array(imagen))
    
    ######### Estado de bg_imagen #########
    if "imagen_escalada" not in st.session_state:
        st.session_state.imagen_escalada = None
    
    # Selección de tipo de escala
    escala_tipo = st.sidebar.selectbox("Selecciona el tipo de escala", ["Aumento de escala", "Redimensionar", "Reducción de escala"])
        
    if escala_tipo == "Aumento de escala":
        escala_factor = st.sidebar.slider("Factor de escala", 0.1, 0.5, 0.25)
        imagen_escalada = rescale(imagen, escala_factor, anti_aliasing=True)
    elif escala_tipo == "Redimensionar":
        img_width = st.sidebar.number_input("Ancho de la imagen", min_value=100, max_value=900, value=min(imagen.shape[1], 900))
        img_height = st.sidebar.number_input("Alto de la imagen", min_value=100, max_value=900, value=min(imagen.shape[0], 900))
        imagen_escalada = resize(imagen, (img_height, img_width), anti_aliasing=True)
    elif escala_tipo == "Reducción de escala":
        downscale_factor = st.sidebar.slider("Factor de downscale", 1, 10, 2)
        imagen_escalada = downscale_local_mean(imagen, (downscale_factor, downscale_factor))
        
    st.session_state.imagen_escalada = (imagen_escalada * 255).astype(np.uint8) # imagen original escalada
    st.image(st.session_state.imagen_escalada, caption="Imagen Original", use_column_width=True)
    
    def actualizar_imagen():
        st.experimental_rerun()
    
######### Estado de Cropping #########
    if "show_cropper" not in st.session_state:
        st.session_state.show_cropper = False
    
    def activar_cropper():
        st.session_state.show_cropper = True
        

    st.sidebar.button("Actualizar imagen", on_click=actualizar_imagen, key="update_image_btn")
    
    st.sidebar.button("Seleccionar región", on_click=activar_cropper, key="select_region_btn")

    
    # SEC-2: Seleccionar región con Cropper
    if st.session_state.imagen_escalada is not None and st.session_state.show_cropper:
        # Convert imagen_escalada (NumPy array) a PIL Image en modo RGB para Crapping
        img_pil = Image.fromarray(st.session_state.imagen_escalada)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # Use st_cropper as a widget for cropping the image
        cropped_img = st_cropper(img_pil)  # This is the correct way to use it
        
        if cropped_img is not None:
            st.image(cropped_img, caption="Región seleccionada", use_column_width=True)

        ######### Rest of the image processing sections #########
        # SEC-3: Contraste
        factor = st.sidebar.slider("Ajustar contraste", 0.5, 3.0, 1.0, key="contrast_slider")
        if st.sidebar.button("Mejorar contraste"):
            region_mejorada = mejorar_contraste(cropped_img, factor)
            st.image(region_mejorada, caption="Región con contraste mejorado", use_column_width=True)
        
        # SEC-4: Contraste con CLAHE
        clahe_clip = st.sidebar.slider("Clip Limit para CLAHE", 0.01, 4.0, 2.0, key="clip_limit")
        clahe_grid = st.sidebar.slider("Tamaño de la cuadrícula para CLAHE", 1, 16, 8, key="grid_size")
        if st.sidebar.button("Aplicar contraste CLAHE"):
            region_clahe = mejorar_contraste_clahe(cropped_img, clahe_clip=clahe_clip, clahe_grid=(clahe_grid, clahe_grid))
            st.image(region_clahe, caption="Región con CLAHE aplicado", use_column_width=True)
        
        # SEC-5: Binarización con OTSU
        if st.sidebar.button("Aplicar binarización de Otsu"):
            region_binarizada = binarizar_otsu(cropped_img)
            st.image(region_binarizada, caption="Región binarizada con Otsu", use_column_width=True)
        
        # SEC-6: Binarización manual
        umbral = st.sidebar.slider("Umbral para binarización manual", 0, 120, 117)
        if st.sidebar.button("Aplicar Otsu manual"):
            region_binarizada_manual = binarizar_manual(cropped_img, umbral)
            st.image(region_binarizada_manual, caption="Región binarizada manualmente", use_column_width=True)
        
        # SEC-7: Segmentación por rango de umbrales
        umbral_min, umbral_max = st.sidebar.slider(
            "Selecciona el rango de umbrales para segmentación",
            0.0, 1.0, (0.4, 0.9),
            step=0.01,
            key="umbral_rango"
        )
        
        if st.sidebar.button("Aplicar segmentación por rango de umbrales"):
            if umbral_min < umbral_max:
                region_segmentada = binarizar_rango(cropped_img, umbral_min, umbral_max)
                st.session_state.region_segmentada = region_segmentada
                st.image(region_segmentada, caption="Región segmentada por rango de umbrales", use_column_width=True)
            else:
                st.error("El umbral mínimo debe ser menor que el umbral máximo.")

        # SEC-8: Segmentación por bordes
        # TODO
        
        # SEC-9: Operadores morfologicos
        # TODO
