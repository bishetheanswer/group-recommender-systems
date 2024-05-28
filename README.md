# Recomendación para grupos
Se han entrenado dos modelos:
* Modelo de predicciones agregadas
* Modelo de modelos agregados

Se ha intentado documentar el código de la forma más clara posible. Si hay algo que
no está claro no duden en contactarme.

## Estructura de la carpeta
```
sistemas-de-recomendacion
 ┣ data/: encontramos el dataset de movielens junto al mapeo de usuarios a grupos
 ┣ models/: encontramos los modelos entrenados
 ┣ Dockerfile
 ┣ README.md
 ┣ aggregation_functions.py: funciones de agregación implementadas para el modelo de predicciones agregadas
 ┣ inference.py: script para realizar inferencia para ambos modelos
 ┣ map_users_to_groups.py: script para mapear los usuarios con los grupos
 ┣ notebook.ipynb: libreta mostrando el proceso de entrenamiento e inferencia
 ┣ notebook_no_functiona.py: código de la libreta si no funcionase
 ┣ requirements.txt: paquetes y versiones usadas
 ┣ train_agg_model.py: script para entrenar un modelo de modelos agregados
 ┗ train_agg_predictions_model.py: script para entrenar un modelo de predicciones agregadas
```

## ¿Cómo se ejecuta?
Contamos con dos alternativas:
### Alternativa 1: Entorno virtual
Para la realización de la práctica se ha usado Python 3.12.1 por lo tanto esta versión
tiene que estar instalada en su ordenador. Se recomienda crear un entorno virtual
para instalar todas las dependencias.

1. Primero instalamos las depenencias
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-use-pep517 surprise
pip install -r requirements.txt
```

2. Probamos a ejecutar los scripts de entrenamiento
```bash
python train_agg_predictions_model.py --test_size 0.3 --top_k 10 --relevance_rating_threshold 3.0 --relevance_min_ratings 2
```
```bash
python train_agg_model.py --test_size 0.3 --top_k 10 --relevance_rating_threshold 3.5
```

### Alternativa 2: Docker
Tenemos que tener docker instalado y movernos a la carpeta de la práctica. Después hacempos:

```bash
docker build -t practica .
docker run -it practica
```
Una vez hecho esto nos encontraremos en el terminal del contenedor desde el cual
podremos ejecutar los scripts con los comandos mostrados en la alternativa 1.