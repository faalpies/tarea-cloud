from sklearn.preprocessing import StandardScaler
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology
from PIL import Image
import PIL
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from tritonclient.utils import *
import tritonclient.http as httpclient
import json

TRITON_SERVER_URL = "triton:8000"
MODEL_NAME = "modelo_svr" 
SCALER_NAME = "scaler"

client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {"jpg", "jpeg"}

def inferencia_triton(cliente, nombre_modelo, datos_entrada):
    entrada = httpclient.InferInput('float_input', datos_entrada.shape, 'FP32')
    entrada.set_data_from_numpy(datos_entrada)

    salida = httpclient.InferRequestedOutput('variable')
    respuesta = cliente.infer(nombre_modelo, [entrada], request_id='0', outputs=[salida])
    return respuesta.as_numpy('variable')

def marcar_bordes_lagos_y_mostrar_rgb_con_bordes(raster_path, tamano_minimo=100):
    # Leer el archivo raster
    with rasterio.open(raster_path) as src:
        # Leer las bandas para NDWI
        verde = src.read(2).astype("float64")
        nir = src.read(4).astype("float64")
        ndwi = (verde - nir) / (verde + nir)

        # Leer las bandas para la imagen RGB
        red = src.read(1).astype("float64")
        green = src.read(2).astype("float64")
        blue = src.read(3).astype("float64")
    # Normalizar las bandas RGB
    red /= red.max()
    green /= green.max()
    blue /= blue.max()

    # Combinar las bandas en una imagen RGB
    rgb_image = np.dstack((red, green, blue))

    # Crear máscara de agua
    agua = (ndwi >= 0.3) & (ndwi <= 1)

    # Etiquetar regiones conectadas y filtrar por tamaño
    etiquetas, _ = ndi.label(agua)
    regiones = measure.regionprops(etiquetas)
    for region in regiones:
        if region.area < tamano_minimo:
            etiquetas[etiquetas == region.label] = 0

    # Crear máscara final con cuerpos de agua del tamaño mínimo
    mascara_final = etiquetas > 0

    # Detectar bordes
    bordes = morphology.dilation(mascara_final) ^ mascara_final

    # Dibujar la imagen RGB con bordes de lagos
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.contour(bordes, colors="red")
    plt.title("Delimitación Agua")
    plt.savefig("image/bordes_lagos.png", format="png")


def imagen_a_dataframe_y_vuelta(ruta_imagen, ruta_salida_tiff):
    # Cargar la imagen y convertirla en un DataFrame
    with Image.open(ruta_imagen) as img:
        imagen_array = np.array(img)
        pixels = imagen_array.reshape(-1, 3)
        df = pd.DataFrame(pixels, columns=["band_1", "band_2", "band_3"])  

    datos_entrada = df.to_numpy().astype(np.float32)
    datos_escalados = inferencia_triton(client, SCALER_NAME, datos_entrada)

    # Agregar la nueva columna de suma de Rojo y Azul
    df["band_5"] = inferencia_triton(client, MODEL_NAME, datos_escalados)

    # Reconvertir el DataFrame en una imagen
    data_array = df.to_numpy()
    imagen_array_modificado = data_array.reshape(
        imagen_array.shape[0], imagen_array.shape[1], 4
    )
    imagen_modificada = Image.fromarray(np.uint8(imagen_array_modificado))

    # Guardar la imagen modificada en formato TIFF
    imagen_modificada.save(ruta_salida_tiff, format="TIFF")
    return ruta_salida_tiff


# Cambiar resolucion de la imagen
def cambiar_resolucion_con_proporcion(ruta_imagen_original, nueva_anchura_o_altura):
    # Cargar la imagen original
    with Image.open(ruta_imagen_original) as img:
        # Obtener las dimensiones originales
        ancho_original, alto_original = img.size

        # Calcular el factor de escala manteniendo la relación de aspecto
        if nueva_anchura_o_altura[0] is not None:  # Si se especifica el ancho
            factor_escala = nueva_anchura_o_altura[0] / ancho_original
            nuevo_ancho = nueva_anchura_o_altura[0]
            nuevo_alto = int(alto_original * factor_escala)
        else:  # Si se especifica el alto
            factor_escala = nueva_anchura_o_altura[1] / alto_original
            nuevo_ancho = int(ancho_original * factor_escala)
            nuevo_alto = nueva_anchura_o_altura[1]

        # Cambiar el tamaño de la imagen
        img_resized = img.resize(
            (nuevo_ancho, nuevo_alto), PIL.Image.Resampling.LANCZOS
        )

        #  imagen redimensionada
        img_resized.save("img_resized.png", format="png")
        return "img_resized.png"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/make-plot", methods=["POST"])
def make_plot():
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Formato no permitido"}), 400

    if file:  # y aquí puedes agregar verificación adicional del tipo de archivo
        filename = secure_filename(file.filename)
        ruta_imagen_original = os.path.join("image/", filename)
        file.save(ruta_imagen_original)

        # Ruta a tu imagen original
        nueva_anchura_o_altura = (250, None)
        # Cambiar la resolución de la imagen manteniendo la proporción
        img_red_ruta = cambiar_resolucion_con_proporcion(
            ruta_imagen_original, nueva_anchura_o_altura
        )

        # Procesar la imagen y guardarla en formato TIFF
        raster = imagen_a_dataframe_y_vuelta(img_red_ruta, "image/bordes_lagos.png")
        marcar_bordes_lagos_y_mostrar_rgb_con_bordes(raster)

        return jsonify({"res": True})
    else:
        return jsonify({"res": False}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")