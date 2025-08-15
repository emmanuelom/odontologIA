import streamlit as st
from streamlit_cropper import st_cropper
from io_img import cargar_imagen, guardar_imagen
from preprocesamiento import mejorar_contraste, seleccionar_region, mejorar_contraste_clahe, binarizar_otsu, binarizar_manual, segmentar_umbral, segmentar_bordes, erosionar, dilatar, binarizar_rango, encontrar_umbral_optimo
from PIL import Image
import numpy as np
from skimage import color
from skimage.transform import resize, rescale, downscale_local_mean
import matplotlib.pyplot as plt

# Configurar matplotlib para que no use GUI
import matplotlib
matplotlib.use('Agg')

# Funciones para manejo del historial
def guardar_estado_actual():
    """Guarda el estado actual de la imagen en el historial"""
    estado = {
        'cropped_img': st.session_state.get('cropped_img'),
        'region_mejorada': st.session_state.get('region_mejorada'),
        'region_clahe': st.session_state.get('region_clahe'),
        'region_binarizada': st.session_state.get('region_binarizada'),
        'region_binarizada_manual': st.session_state.get('region_binarizada_manual'),
        'region_segmentada': st.session_state.get('region_segmentada'),
        'region_bordes': st.session_state.get('region_bordes'),
        'region_erosionada': st.session_state.get('region_erosionada'),
        'region_dilatada': st.session_state.get('region_dilatada'),
        'region_umbral_optimo': st.session_state.get('region_umbral_optimo'),
        'umbral_optimo': st.session_state.get('umbral_optimo'),
        'histograma_fig': st.session_state.get('histograma_fig')
    }
    if 'historial_imagenes' not in st.session_state:
        st.session_state.historial_imagenes = []
    st.session_state.historial_imagenes.append(estado)

def deshacer_ultimo_cambio():
    """Deshace el último cambio realizado"""
    if 'historial_imagenes' in st.session_state and len(st.session_state.historial_imagenes) > 0:
        estado_anterior = st.session_state.historial_imagenes.pop()
        for key, value in estado_anterior.items():
            st.session_state[key] = value
        st.session_state.ultima_accion = "Deshecho"
        st.success("✅ Último cambio deshecho exitosamente")
    else:
        st.warning("⚠️ No hay cambios para deshacer")

def reiniciar_aplicacion():
    """Reinicia la aplicación eliminando todas las imágenes"""
    keys_to_reset = [
        'cropped_img', 'region_mejorada', 'region_clahe', 'region_binarizada',
        'region_binarizada_manual', 'region_segmentada', 'region_bordes',
        'region_erosionada', 'region_dilatada', 'region_umbral_optimo',
        'umbral_optimo', 'histograma_fig', 'show_cropper', 'historial_imagenes'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.ultima_accion = "Reiniciado"
    st.success("🔄 Aplicación reiniciada exitosamente")

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
        background-color: #45a049;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: transparent;
        color: #45a049;
        border: 2px solid #45a049;
    }
    .stTitle {
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
    }
    .center {
        display: flex;
        justify-content: center;
    }
    /* Estilos especiales para botones de control */
    div[data-testid="column"]:first-child .stButton > button {
        background-color: #f39c12;
        border: 2px solid #e67e22;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="column"]:first-child .stButton > button:hover {
        background-color: #e67e22;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background-color: #e74c3c;
        border: 2px solid #c0392b;
        justify-content: center;
        align-items: center;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button:hover {
        background-color: #c0392b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Inicio
# --- Botones de control en la parte superior izquierda ---
top_col1, top_col2 = st.columns([4, 1])
with top_col1:
    if st.button("↶ Deshacer", help="Deshace el último cambio realizado", key="undo_btn"):
        deshacer_ultimo_cambio()
with top_col2:
    if st.button("🔄 Reiniciar", help="Reinicia la aplicación y elimina todas las imágenes", key="reset_btn"):
        reiniciar_aplicacion()

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

#region ESTADOS DE SESIÓN
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
if 'umbral_optimo' not in st.session_state:
    st.session_state.umbral_optimo = None
if 'region_umbral_optimo' not in st.session_state:
    st.session_state.region_umbral_optimo = None
if 'histograma_fig' not in st.session_state:
    st.session_state.histograma_fig = None
if 'historial_imagenes' not in st.session_state:
    st.session_state.historial_imagenes = []
if 'ultima_accion' not in st.session_state:
    st.session_state.ultima_accion = None

#endregion


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
        escala_tipo = st.selectbox("Tamaño de Imagen", ["Agrandar imagen", "Redimensionar", "Reducir imagen"])
        
        if escala_tipo == "Agrandar imagen":
            escala_factor = st.slider("Factor de escala", 0.1, 0.5, 0.25)
            imagen_escalada = rescale(imagen, escala_factor, anti_aliasing=True)
        elif escala_tipo == "Redimensionar":
            img_width = st.number_input("Ancho de la imagen", min_value=100, max_value=900, value=min(imagen.shape[1], 900))
            img_height = st.number_input("Alto de la imagen", min_value=100, max_value=900, value=min(imagen.shape[0], 900))
            imagen_escalada = resize(imagen, (img_height, img_width), anti_aliasing=True)
        elif escala_tipo == "Reducir imagen":
            downscale_factor = st.slider("Factor de downscale", 1, 10, 2)
            imagen_escalada = downscale_local_mean(imagen, (downscale_factor, downscale_factor))
        
        st.session_state.imagen_escalada = (imagen_escalada * 255).astype(np.uint8) # imagen original escalada
    
    # Mostrar imagen original en un expander
    with st.expander("📷 Ver imagen original"):
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
            # Mostrar región seleccionada en un expander
            with st.expander("✂️ Ver región seleccionada"):
                st.image(cropped_img, caption="Región seleccionada", use_column_width=True)

        #endregion

        #region REALCE
        with tab_realce:
            st.image(st.session_state.cropped_img, caption="Región seleccionada", use_column_width=True)
            
            # Botón para encontrar umbral óptimo
            if st.button("🎯 Encontrar umbral óptimo"):
                guardar_estado_actual()  # Guardar estado antes del cambio
                umbral_optimo, region_umbral_optimo, histograma_fig = encontrar_umbral_optimo(st.session_state.cropped_img)
                st.session_state.umbral_optimo = umbral_optimo
                st.session_state.region_umbral_optimo = region_umbral_optimo
                st.session_state.histograma_fig = histograma_fig
                st.success(f"Umbral óptimo encontrado: {umbral_optimo}")
            
            # Mostrar histograma en sidebar si existe
            if st.session_state.histograma_fig is not None:
                st.sidebar.markdown("### 📊 Análisis de umbral")
                st.sidebar.pyplot(st.session_state.histograma_fig, use_container_width=True)
            
            # Ajustes avanzados de contraste en expander
            with st.expander("⚙️ Ajustes avanzados de contraste"):
                # SEC-3: Contraste
                factor = st.slider("Ajustar contraste manualmente", 0.5, 3.0, 1.0, key="contrast_slider")
                if st.button("Mejorar contraste"):
                    guardar_estado_actual()  # Guardar estado antes del cambio
                    region_mejorada = mejorar_contraste(st.session_state.cropped_img, factor)
                    st.session_state.region_mejorada = region_mejorada
            
                # SEC-4: Contraste con CLAHE
                st.markdown("### Mejora automática del contraste")
                clahe_clip = st.slider("Intensidad", 0.01, 4.0, 2.0, key="clip_limit")
                clahe_grid = st.slider("Detalle", 1, 16, 8, key="grid_size")
                if st.button("Aplicar contraste"):
                    guardar_estado_actual()  # Guardar estado antes del cambio
                    region_clahe = mejorar_contraste_clahe(st.session_state.cropped_img, clahe_clip=clahe_clip, clahe_grid=(clahe_grid, clahe_grid))
                    st.session_state.region_clahe = region_clahe
        
        #endregion
        
    # Mostrar resultado del umbral óptimo si existe
    if st.session_state.region_umbral_optimo is not None:
        with st.expander("🎯 Ver imagen con binarización óptima"):
            st.image(st.session_state.region_umbral_optimo, caption=f"Binarización con umbral óptimo ({st.session_state.umbral_optimo})", use_column_width=True)
        
    # Mostrar contraste mejorado en expander si existe
    if st.session_state.region_mejorada is not None:
        with st.expander("🔆 Ver imagen con contraste mejorado"):
            st.image(st.session_state.region_mejorada, caption="Región con contraste mejorado", use_column_width=True)
    
    # Mostrar CLAHE en expander si existe
    if st.session_state.region_clahe is not None:
        with st.expander("📈 Ver imagen con CLAHE aplicado"):
            st.image(st.session_state.region_clahe, caption="Región con CLAHE aplicado", use_column_width=True)
    
    # Solo mostrar la pestaña de filtro si hay una región seleccionada
    if st.session_state.cropped_img is not None:
        #region FILTRO
        with tab_filtro:
            st.image(st.session_state.cropped_img, caption="Región seleccionada", use_column_width=True)
            # SEC-5: Binarización con OTSU
            if st.button("Identificar blancos y negros"):
                guardar_estado_actual()  # Guardar estado antes del cambio
                region_binarizada = binarizar_otsu(st.session_state.cropped_img)
                st.session_state.region_binarizada = region_binarizada
        
            # Ajustes avanzados de filtrado en expander
            with st.expander("⚙️ Ajustes avanzados de filtrado"):
                # SEC-6: Binarización manual
                st.markdown("### Binarización manual")
                umbral = st.slider("Umbral para binarización manual", 0, 120, 117)
                if st.button("Aplicar Otsu manual"):
                    region_binarizada_manual = binarizar_manual(st.session_state.cropped_img, umbral)
                    st.session_state.region_binarizada_manual = region_binarizada_manual
            
                # SEC-7: Segmentación por rango de umbrales
                st.markdown("### Resaltar por rango")
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
                st.markdown("### Detección de bordes")
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
            with st.expander("🔧 Bordes y Áreas"):
                st.markdown("### Operadores morfológicos")
                tipo_segmentacion = st.selectbox("Selecciona el tipo de segmentación para aplicar operadores morfológicos", ["Umbrales", "Bordes"])
                
                # EROSIÓN
                if st.button("Reducir detalles"):
                    if tipo_segmentacion == "Umbrales" and st.session_state.region_segmentada is not None:
                        region_erosionada = erosionar(st.session_state.region_segmentada)
                        st.session_state.region_erosionada = region_erosionada
                        st.session_state.tipo_erosion = "Umbrales" 
                    elif tipo_segmentacion == "Bordes" and st.session_state.region_bordes is not None:
                        region_erosionada = erosionar(st.session_state.region_bordes)
                        st.session_state.region_erosionada = region_erosionada
                        st.session_state.tipo_erosion = "Bordes"  
                    
                # DILATACIÓN
                if st.button("Resaltar detalles"):
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
        
    # Mostrar imagen binarizada con Otsu como resultado principal
    if st.session_state.region_binarizada is not None:
        with st.expander("🔲 Ver imagen binarizada con Otsu"):
            st.image(st.session_state.region_binarizada, caption="Región binarizada con Otsu", use_column_width=True)

    # Mostrar otras imágenes en expanders para no saturar la vista
    if st.session_state.region_binarizada_manual is not None:
        with st.expander("🔧 Ver binarización manual"):
            st.image(st.session_state.region_binarizada_manual, caption="Región binarizada manualmente", use_column_width=True)

    if st.session_state.region_segmentada is not None:
        with st.expander("📊 Ver segmentación por rango de umbrales"):
            st.image(st.session_state.region_segmentada, caption="Región segmentada por rango de umbrales", use_column_width=True)

    if st.session_state.region_bordes is not None:
        with st.expander("🔍 Ver detección de bordes"):
            st.image(st.session_state.region_bordes, caption="Región segmentada por bordes", use_column_width=True)

    # Mostrar resultados de operadores morfológicos como resultados principales si existen
    if st.session_state.region_erosionada is not None:
        tipo = st.session_state.tipo_erosion if "tipo_erosion" in st.session_state else "Desconocido"
        with st.expander(f"⚫ Ver imagen con erosión ({tipo})"):
            st.image(st.session_state.region_erosionada, caption=f"Región erosionada ({tipo})", use_column_width=True)

    if st.session_state.region_dilatada is not None:
        tipo = st.session_state.tipo_dilatacion if "tipo_dilatacion" in st.session_state else "Desconocido"
        with st.expander(f"⚪ Ver imagen con dilatación ({tipo})"):
            st.image(st.session_state.region_dilatada, caption=f"Región dilatada ({tipo})", use_column_width=True)
        