import streamlit as st 
from streamlit_drawable_canvas import st_canvas
from io_img import cargar_imagen, guardar_imagen
from preprocesamiento import mejorar_contraste, seleccionar_region, mejorar_contraste_clahe, mejorar_enfoque
from PIL import Image
import numpy as np

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
        color: #4CAF50;
    }
    .stMarkdown {
        text-align: center;
    </style>
    """,
    unsafe_allow_html=True
)


# Inicio
st.markdown("<h1 class='stTitle'>Bienvenido a odontologIA</h1>", unsafe_allow_html=True)
st.markdown("<p class='stMarkdown'>Carga una imagen para comenzar</p>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <h1 style='font-size: 50px;'>
        odontolog<span style='color: #4CAF50;'>IA</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# SEC-1: Cargar una imagen
imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], key="file_uploader")
if imagen_subida is not None:
    imagen = cargar_imagen(imagen_subida)
    
    # Conversión array a img PIL
    if isinstance(imagen, np.ndarray):
        imagen = Image.fromarray(imagen)
    
    # Escala de imagen
    img_width = imagen.width
    img_height = imagen.height
    default_width = min(img_width, 800)
    default_height = min(img_height, 400)
    
    img_width = st.sidebar.number_input("Ancho de la imagen", min_value=100, max_value=800, value=default_width)
    img_height = st.sidebar.number_input("Alto de la imagen", min_value=100, max_value=800, value=default_height)
    
    imagen = imagen.resize((img_width, img_height))
    
    st.image(imagen, caption="Imagen cargada", use_column_width=True)  # versión streamlit +1.25 usa use_container_width
    
    # SEC-2: Seleccionar regióN
    st.sidebar.markdown("### Seleccionar región")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno con transparencia
        stroke_width=2,
        background_image=imagen,
        update_streamlit=True,
        height=img_height,
        width=img_width,
        drawing_mode="rect",
        key="canvas",
    )
    
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "rect":
                left = int(obj["left"])
                top = int(obj["top"])
                width = int(obj["width"])
                height = int(obj["height"])
                right = left + width
                bottom = top + height
                region = imagen.crop((left, top, right, bottom))
                st.image(region, caption="Región seleccionada", use_column_width=True)
                
                # SEC-3: Contraste
                factor = st.sidebar.slider("Ajustar contraste", 0.5, 3.0, 1.0, key="contrast_slider")
                if st.sidebar.button("Mejorar contraste"):
                    region_mejorada = mejorar_contraste(region, factor)
                    st.image(region_mejorada, caption="Región con contraste mejorado", use_column_width=True)
                
                # SEC-4: Contraste con CLAHE
                clahe_clip = st.sidebar.slider("Clip Limit para CLAHE", 0.01, 4.0, 2.0, key="clip_limit")
                # CLAHE GRID: Un valor más pequeño puede mejorar el contraste en áreas pequeñas
                clahe_grid = st.sidebar.slider("Tamaño de la cuadrícula para CLAHE", 1, 16, 8, key="grid_size")
                if st.sidebar.button("Aplicar contraste CLAHE"):
                    region_clahe = mejorar_contraste_clahe(region, clahe_clip=clahe_clip, clahe_grid=(clahe_grid, clahe_grid))
                    st.image(region_clahe, caption="Región con CLAHE aplicado", use_column_width=True)
                
                # SEC-5: Enfoque con rolling_ball
                focus_radius = st.sidebar.slider("Radio para rolling_ball", 10, 200, 50, key="rolling_ball_radius")
                if st.sidebar.button("Aplicar enfoque"):
                    region_enfocada = mejorar_enfoque(region, focus_radius=focus_radius)
                    st.image(region_enfocada, caption="Región con enfoque aplicado", use_column_width=True)
    # Agregaremos funcionamiento a estos botones luego
### st.sidebar.button("Segmentar imagen")
### st.sidebar.button("Segmentar umbrales")