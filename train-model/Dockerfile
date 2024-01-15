#  Python como base
FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*


ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal


RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

RUN pip install --no-cache-dir -r librerias.txt

CMD ["tail", "-f", "/dev/null"]