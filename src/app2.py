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
col1, col2, col3 = st.columns([2,3,2])
with col2:
    st.image("data/lasalleuni.png", width=320)
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
if 'region_binarizada_manual' not in st.session_state:
    st.session_state.region_binarizada_manual = None
if 'region_bordes' not in st.session_state:
    st.session_state.region_bordes = None
if 'imagen_escalada' not in st.session_state:
    st.session_state.imagen_escalada = None
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'show_cropper' not in st.session_state:
    st.session_state.show_cropper = False
if 'region_segmentada' not in st.session_state:
    st.session_state.region_segmentada = None
if 'region_mejorada' not in st.session_state:
    st.session_state.region_mejorada = None
if 'region_clahe' not in st.session_state:
    st.session_state.region_clahe = None
if 'region_erosionada' not in st.session_state:
    st.session_state.region_erosionada = None
if 'region_dilatada' not in st.session_state:
    st.session_state.region_dilatada = None
if 'region_bordes' not in st.session_state:
    st.session_state.region_bordes = None


## Pestañas en Sidebar: Imagen(Image), Realce(Enhancement), Filto(Filter)
tab_imagen, tab_realce, tab_filtro = st.sidebar.tabs(["Imagen", "Realce", "Filtro"])

#region IMAGEN 
# SEC-1: Cargar una imagen
with tab_imagen:
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="file_uploader")

if imagen_subida is not None:
    imagen = cargar_imagen(imagen_subida)
    
    # Conversión img skimage array a img PIL
    if isinstance(imagen, np.ndarray):
        imagen = Image.fromarray(imagen)
    imagen = color.rgb2gray(np.array(imagen))
    
    # Selección de tipo de escala
    with tab_imagen:
        escala_tipo = st.selectbox("Selecciona el tipo de escala", ["Aumento de escala", "Redimensionar", "Reducción de escala"])
        
        if escala_tipo == "Aumento de escala":
            escala_factor = st.slider("Factor de escala", 0.1, 0.5, 0.25)
            imagen_escalada = rescale(imagen, escala_factor, anti_aliasing=True)
        elif escala_tipo == "Redimensionar":
            img_width = st.number_input("Ancho de la imagen", min_value=100, max_value=900, value=min(imagen.shape[1], 900))
            img_height = st.number_input("Alto de la imagen", min_value=100, max_value=900, value=min(imagen.shape[0], 900))
            imagen_escalada = resize(imagen, (img_height, img_width), anti_aliasing=True)
        elif escala_tipo == "Reducción de escala":
            downscale_factor = st.slider("Factor de downscale", 1, 10, 2)
            imagen_escalada = downscale_local_mean(imagen, (downscale_factor, downscale_factor))
        
        st.session_state.imagen_escalada = (imagen_escalada * 255).astype(np.uint8) # imagen original escalada
    
    st.image(st.session_state.imagen_escalada, caption="Imagen Original", use_column_width=True)
    
    def actualizar_imagen():
        st.experimental_rerun()
    
    def activar_cropper():
        st.session_state.show_cropper = True
    
    with tab_imagen:
        st.button("Actualizar imagen", on_click=actualizar_imagen, key="update_image_btn")
        st.button("Seleccionar región", on_click=activar_cropper, key="select_region_btn")
    
    # SEC-2: Seleccionar región con Cropper
    if st.session_state.imagen_escalada is not None and st.session_state.show_cropper:
        # Convert imagen_escalada (NumPy array) a PIL Image en modo RGB para Crapping
        img_arr = st.session_state.imagen_escalada
        if img_arr.ndim == 2:  # Si es una imagen en escala de grises
            img_arr = np.stack((img_arr,) * 3, axis=-1)  # Convertir a RGB
        img_pil = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')

        # Use st_cropper as a widget for cropping the image
        cropped_img = st_cropper(img_pil)  # This is the correct way to use it
        if cropped_img is not None:
            st.session_state.cropped_img = cropped_img  # Guardar la imagen recortada en el estado de sesión
            st.image(cropped_img, caption="Región seleccionada", use_column_width=True)

        #endregion

        #region REALCE
        with tab_realce:
            st.image(st.session_state.cropped_img, caption="Región seleccionada", use_column_width=True)
            # SEC-3: Contraste
            factor = st.slider("Ajustar contraste", 0.5, 3.0, 1.0, key="contrast_slider")
            if st.button("Mejorar contraste"):
                region_mejorada = mejorar_contraste(st.session_state.cropped_img, factor)
                st.session_state.region_mejorada = region_mejorada
        
            # SEC-4: Contraste con CLAHE
            st.markdown("### Mejora de contraste con CLAHE")
            clahe_clip = st.slider("Clip Limit para CLAHE", 0.01, 4.0, 2.0, key="clip_limit")
            clahe_grid = st.slider("Tamaño de la cuadrícula para CLAHE", 1, 16, 8, key="grid_size")
            if st.button("Aplicar contraste CLAHE"):
                region_clahe = mejorar_contraste_clahe(st.session_state.cropped_img, clahe_clip=clahe_clip, clahe_grid=(clahe_grid, clahe_grid))
                st.session_state.region_clahe = region_clahe
        
        #endregion
        
    if st.session_state.region_mejorada is not None:
        st.image(st.session_state.region_mejorada, caption="Región con contraste mejorado", use_column_width=True)
    
    if st.session_state.region_clahe is not None:
        st.image(st.session_state.region_clahe, caption="Región con CLAHE aplicado", use_column_width=True)
        
        #region FILTRO
        with tab_filtro:
            st.image(st.session_state.cropped_img, caption="Región seleccionada", use_column_width=True)
            # SEC-5: Binarización con OTSU
            if st.button("Aplicar binarización de Otsu"):
                region_binarizada = binarizar_otsu(st.session_state.cropped_img)
                st.session_state.region_binarizada = region_binarizada
        
            # SEC-6: Binarización manual
            st.markdown("### Binarización manual")
            umbral = st.slider("Umbral para binarización manual", 0, 120, 117)
            if st.button("Aplicar Otsu manual"):
                region_binarizada_manual = binarizar_manual(st.session_state.cropped_img, umbral)
                st.session_state.region_binarizada_manual = region_binarizada_manual
        
            # SEC-7: Segmentación por rango de umbrales
            st.markdown("### Segmentación por rango de umbrales")
            umbral_min, umbral_max = st.slider(
                "Selecciona el rango de umbrales para segmentación",
                0.0, 1.0, (0.4, 0.9),
                step=0.01,
                key="umbral_rango"
            )
        
            if st.button("Aplicar segmentación por rango de umbrales"):
                if umbral_min < umbral_max:
                    region_segmentada = binarizar_rango(st.session_state.cropped_img, umbral_min, umbral_max)
                    st.session_state.region_segmentada = region_segmentada
                else:
                    st.error("El umbral mínimo debe ser menor que el umbral máximo.")
            
            # SEC-8: Segmentación por bordes
            st.markdown("### Segmentación por bordes")
            sigma_bordes = st.slider("Sigma para detección de bordes", 0.1, 5.0, 1.0)
            if st.button("Aplicar segmentación por bordes"):
                # Convertir a escala de grises si es necesario
                img_bordes = np.array(st.session_state.cropped_img)
                if img_bordes.ndim == 3:
                    from skimage.color import rgb2gray
                    img_bordes = rgb2gray(img_bordes)
                region_bordes = segmentar_bordes(img_bordes, sigma=sigma_bordes)
                region_bordes = (region_bordes * 255).astype(np.uint8)
                st.session_state.region_bordes = region_bordes
            
            # SEC-9: Operadores morfológicos
            st.markdown("### Operadores morfológicos")
            tipo_segmentacion = st.selectbox("Selecciona el tipo de segmentación para aplicar operadores morfológicos", ["Umbrales", "Bordes"])
            
            # EROSIÓN
            if st.button("Aplicar erosión"):
                if tipo_segmentacion == "Umbrales" and st.session_state.region_segmentada is not None:
                    region_erosionada = erosionar(st.session_state.region_segmentada)
                    st.session_state.region_erosionada = region_erosionada
                    st.session_state.tipo_erosion = "Umbrales" 
                elif tipo_segmentacion == "Bordes" and st.session_state.region_bordes is not None:
                    region_erosionada = erosionar(st.session_state.region_bordes)
                    st.session_state.region_erosionada = region_erosionada
                    st.session_state.tipo_erosion = "Bordes"  
                
            # DILATACIÓN
            if st.button("Aplicar dilatación"):
                if tipo_segmentacion == "Umbrales" and st.session_state.region_segmentada is not None:
                    region_dilatada = dilatar(st.session_state.region_segmentada)
                    st.session_state.region_dilatada = region_dilatada
                    st.session_state.tipo_dilatacion = "Umbrales"  
                elif tipo_segmentacion == "Bordes" and st.session_state.region_bordes is not None:
                    region_dilatada = dilatar(st.session_state.region_bordes)
                    st.session_state.region_dilatada = region_dilatada
                    st.session_state.tipo_dilatacion = "Bordes" 
                else:
                    st.warning("Primero aplica la segmentación seleccionada")

        #endregion
        
    if st.session_state.region_binarizada is not None:
        st.image(st.session_state.region_binarizada, caption="Región binarizada con Otsu", use_column_width=True)

    if st.session_state.region_binarizada_manual is not None:
        st.image(st.session_state.region_binarizada_manual, caption="Región binarizada manualmente", use_column_width=True)

    if st.session_state.region_segmentada is not None:
        st.image(st.session_state.region_segmentada, caption="Región segmentada por rango de umbrales", use_column_width=True)

    if st.session_state.region_bordes is not None:
        st.image(st.session_state.region_bordes, caption="Región segmentada por bordes", use_column_width=True)

    if st.session_state.region_erosionada is not None:
        tipo = st.session_state.tipo_erosion if "tipo_erosion" in st.session_state else "Desconocido"
        st.image(st.session_state.region_erosionada, caption=f"Región erosionada ({tipo})", use_column_width=True)

    if st.session_state.region_dilatada is not None:
        tipo = st.session_state.tipo_dilatacion if "tipo_dilatacion" in st.session_state else "Desconocido"
        st.image(st.session_state.region_dilatada, caption=f"Región dilatada ({tipo})", use_column_width=True)
        