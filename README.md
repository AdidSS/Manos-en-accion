# Manos en acción
Reconocimiento de Lenguaje de Señas con Aprendizaje profundo


Victor Adid Salgado Santana A01710023

| **Archivo**   | **Descripción**                                              |
|---------------|--------------------------------------------------------------|
| get_signs_loop.py     | Código para tomar fotos de cada seña y armar el dataset propio de LSM          |
| manos_en_accion_segmentacion_modelo.ipynb     | Entrenamiento del modelo yolov8 para segmentar manos (dataset de manos obtenido de roboflow)|
| manos_en_accion_ETL.ipynb  | Extracción de datasets, segmentación de manos y guardado de imágenes (manos y fondo negro)   |
| manos__en_accion_partial_unfreezing.ipynb  | Descongelamiento progresivo de capas y evaluación del desempeño |
| manos__en_accion_modeling.ipynb   | Entrenamiento del modelo final |
| real_time_predictions.py   | Clasificaciones en tiempo real con pipeline de segmentación (yolov8) y clasificación (efficientnet b0)|
| Manos_en_acción_Reconocimiento_del_lenguaje_de_señas.pdf| Reporte Final |
