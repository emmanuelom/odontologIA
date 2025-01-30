import streamlit as st 

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
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #6A8EAE;
        color: white;
        border: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Inicio
st.title("Bienvenido a odontologIA")
st.write("Carga una imagen para comenzar")

st.sidebar.header("odontologIA")

# SEC-1: Cargar una imagen
st.sidebar.button("Cargar imagen")

# SEC-2: Región, mejoras y segmentación
st.sidebar.button("Seleccionar región")
st.sidebar.button("Contraste")
st.sidebar.button("Segmentar imagen")
st.sidebar.button("Segmentar umbrales")
