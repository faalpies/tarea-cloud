import os
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx


def raster_to_dataframe(raster_path):
    with rasterio.open(raster_path) as raster:
        # Leer las bandas del raster
        bands = raster.read()

        # Obtener dimensiones
        height, width = bands.shape[1], bands.shape[2]

        # Preparar datos para el DataFrame
        data = {
            "filename_pixel": [],
            "band_1": [],
            "band_2": [],
            "band_3": [],
            "band_4": [],
            "band_5": [],
        }

        # Extraer la parte relevante del nombre del archivo
        filename = (
            raster_path.split("/")[-1]
            .replace("satellite_image-", "")
            .replace(".tif", "")
        )

        # Llenar los datos
        for y in range(height):
            for x in range(width):
                pixel_label = f"{filename}_x{x}_y{y}"
                data["filename_pixel"].append(pixel_label)
                for i in range(bands.shape[0]):
                    data[f"band_{i+1}"].append(bands[i, y, x])
                if bands[1, y, x] == 0 and bands[3, y, x] == 0:
                    band_5_value = 0
                else:
                    band_5_value = (bands[1, y, x] - bands[3, y, x]) / (
                        bands[1, y, x] + bands[3, y, x]
                    )
                data["band_5"].append(band_5_value)
        # Crear DataFrame
        df = pd.DataFrame(data)

    return df


# Directorio que contiene los archivos TIFF
directory = "images"

# Lista para almacenar DataFrames de cada archivo
df_list = []

# Leer cada archivo TIFF en el directorio y agregar al DataFrame
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        raster_path = os.path.join(directory, filename)
        df = raster_to_dataframe(raster_path)
        df_list.append(df)

# Combinar todos los DataFrames en uno solo
combined_df = pd.concat(df_list, ignore_index=True)

combined_df["band_5"].replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df["band_5"].fillna(900, inplace=True)


X = combined_df[["band_1", "band_2", "band_3"]]  # Características
y = combined_df["band_5"]  # Etiqueta

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_onnx = to_onnx(scaler, initial_types=[('float_input', FloatTensorType([None, X_train.shape[1]]))])

scaler_model_name = "models/scaler/1/scaler.onnx"
with open(scaler_model_name, "wb") as f:
    f.write(scaler_onnx.SerializeToString())

print(f"Scaler exportado a ONNX en {scaler_model_name}")

# Entrenar el modelo SVM
svm_model = SVR()

print("Entrenando...")
svm_model.fit(X_train_scaled, y_train)

# Predecir y evaluar
print("Evaluando...")
y_pred = svm_model.predict(X_test_scaled)
mse = round(mean_squared_error(y_test, y_pred), 2)
print(f"Error cuadrático medio (MSE): {mse}")

# Convertir a ONNX
initial_type = [("float_input", FloatTensorType([None, X_train_scaled.shape[1]]))]
onnx_model = convert_sklearn(svm_model, initial_types=initial_type)

# Exportar el modelo ONNX
onnx_model_name = "models/modelo_svr/1/" + "modelo_svr.onnx"
with open(onnx_model_name, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Modelo exportado a ONNX en {onnx_model_name}")