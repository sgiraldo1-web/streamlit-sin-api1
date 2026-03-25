import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Malaria",
    page_icon="🦠",
    layout="centered"
)

CLASS_NAMES = ["Uninfected", "Parasitized"]

@st.cache_resource
def load_model():
    st.write("Cargando artifacts/lenet.h5")
    return tf.keras.models.load_model("artifacts/lenet.h5")

def get_model_input_size(model):
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(f"Forma de entrada no soportada: {input_shape}")

    _, height, width, channels = input_shape

    if height is None or width is None:
        raise ValueError(f"No se pudo inferir tamaño de entrada del modelo: {input_shape}")

    return int(height), int(width), int(channels)

def preprocess_image(image: Image.Image, model) -> np.ndarray:
    height, width, channels = get_model_input_size(model)

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((width, height))
    img_array = np.array(image, dtype=np.float32) / 255.0

    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image: Image.Image):
    processed = preprocess_image(image, model)
    prediction = model.predict(processed, verbose=0)

    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        label = CLASS_NAMES[0] if score >= 0.5 else CLASS_NAMES[1]
        confidence = score if score >= 0.5 else 1 - score
    else:
        pred_idx = int(np.argmax(prediction, axis=1)[0])
        label = CLASS_NAMES[pred_idx]
        confidence = float(np.max(prediction))

    return label, confidence * 100, prediction.shape, processed.shape

st.title("🦠 Clasificador de Malaria")
st.markdown(
    """
    Esta aplicación permite cargar una imagen de una célula sanguínea y clasificarla
    con un modelo CNN tipo LeNet.

    **Clases:**
    - **Parasitized**
    - **Uninfected**
    """
)

try:
    model = load_model()
    input_shape = model.input_shape
    output_shape = model.output_shape
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

st.write("**Input shape del modelo:**", input_shape)
st.write("**Output shape del modelo:**", output_shape)

uploaded = st.file_uploader(
    "Sube una imagen de célula (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        img = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Imagen cargada", use_container_width=True)

        with col2:
            st.write("### Información")
            st.write(f"**Nombre:** {uploaded.name}")
            st.write(f"**Tipo:** {uploaded.type}")
            st.write(f"**Tamaño:** {round(uploaded.size / 1024, 2)} KB")

        if st.button("Predecir", use_container_width=True):
            with st.spinner("Procesando imagen..."):
                label, confidence, prediction_shape, processed_shape = predict_image(model, img)

            st.write("## Resultado")

            if label == "Parasitized":
                st.error(f"**Clase predicha:** {label}")
            else:
                st.success(f"**Clase predicha:** {label}")

            st.metric("Confianza", f"{confidence:.2f}%")
            st.write(f"**Forma de entrada enviada al modelo:** {processed_shape}")
            st.write(f"**Forma de salida del modelo:** {prediction_shape}")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")

st.markdown("---")
st.caption("Aplicación desplegada con Streamlit.")
