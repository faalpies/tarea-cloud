##  Cloud Computing
Profesor: Cristhian Aguilera
Alumna: Fabiola Pinto

El modelo, se entrena a patir de una imagen satelital Sentinel-2, en donde se obtiene el indice NDWI (Normalized Difference Water Index) que se utiliza para resaltar  las masas de agua, este indice corresponde al valor a predecir. Con este valor y las bandas que capturan el rojo, verde y azul se entrena el modelo para poder predecir el indice NDWI a partir de cualquier imagen que se tome desde google earth en formato jpeg. De esta manera el modelo remarca con una linea roja el contorno donde hay masas de agua en la captura de imagen.

Se necesita Docker y Docker compose

Para iniciar el proyecto de forma local se debe realizar lo siguiente:
1. Abrir el terminal y obtener la carpeta raíz del proyecto.
2. Ejecute el comando `docker-compose up -d`. Se iniciarán cuatro contenedores en el sistema (backend, training, frontend y triton)
3. Cuando los contenedores estén activos, se podrá acceder a la aplicación a través de su navegador web en la dirección http://localhost:8052.
4. Para entrenar el modelo, se debe ejecutar el siguiente comando en la misma terminal: `docker-compose exec train-model python train.py`. Se generará y actualizarán los archivos del modelo.
