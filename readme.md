##  Cloud Computing
Profesor: Cristhian Aguilera
Alumna: Fabiola Pinto

Se necesita Docker y Docker compose

Para iniciar el proyecto se debe realizar lo siguiente:
1. Abrir el terminal y obtener la carpeta raíz del proyecto.
2. Ejecute el comando `docker-compose up -d`. Se iniciarán tres contenedores en el sistema.
3. Cuando los contenedores estén activos, se podrá acceder a la aplicación a través de su navegador web en la dirección http://localhost:8052.
4. Para entrenar el modelo, se debe ejecutar el siguiente comando en la misma terminal: `docker-compose exec training python train.py`. Se generará y actualizarán los archivos del modelo.