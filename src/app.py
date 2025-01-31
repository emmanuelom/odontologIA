import streamlit as st 
from streamlit_drawable_canvas import st_canvas
from io_img import cargar_imagen, guardar_imagen
from preprocesamiento import mejorar_contraste, seleccionar_region

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
    
    # Redimensionar la imagen si es muy grande
    max_width = 700
    max_height = 700
    if imagen.width > max_width or imagen.height > max_height:
        imagen.thumbnail((max_width, max_height))
    
    st.image(imagen, caption="Imagen cargada", use_column_width=True)
    
    # SEC-2: Seleccionar región
    st.sidebar.markdown("### Seleccionar región")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Color de relleno con transparencia
        stroke_width=2,
        background_image=imagen,
        update_streamlit=True,
        height=imagen.height,
        width=imagen.width,
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
                    
    # Agregaremos funcionamiento a estos botones luego
### st.sidebar.button("Segmentar imagen")
### st.sidebar.button("Segmentar umbrales")