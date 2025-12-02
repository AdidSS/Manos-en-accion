# Manos en acción
Reconocimiento de Lenguaje de Señas con Aprendizaje profundo


Victor Adid Salgado Santana A01710023
![Presentando Manos en Acción](assets/logo.png)

## Descripción de repositorio
| **Archivo**   | **Descripción**                                              |
|---------------|--------------------------------------------------------------|
| get_signs_loop.py     | Código para tomar fotos de cada seña y armar el dataset propio de LSM          |
| manos_en_accion_segmentacion_modelo.ipynb     | Entrenamiento del modelo yolov8 para segmentar manos (dataset de manos obtenido de roboflow)|
| manos_en_accion_ETL.ipynb  | Extracción de datasets, segmentación de manos y guardado de imágenes (manos y fondo negro)   |
| manos__en_accion_partial_unfreezing.ipynb  | Descongelamiento progresivo de capas y evaluación del desempeño |
| manos__en_accion_modeling.ipynb   | Entrenamiento del modelo final |
| manos_en_accion_predictions.ipynb   | Evaluación de las clasificaciones del modelo final|
| real_time_predictions.py   | Clasificaciones en tiempo real con pipeline de segmentación (yolov8) y clasificación (efficientnet b0)|
| Manos_en_acción_Reconocimiento_de_lengua_de_señas.pdf| Reporte Final |
| best_seg.pt| Pesos del modelo de segmentación Yolo v8 |
| efficientNet_final_pro.pth | Pesos del modelo de clasificación EfficientNet B0 |

## Instalación y Ejecución

Para ejecutar este proyecto en tu entorno local, sigue estos pasos:

### Prerrequisitos

- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Configuración del entorno

1. Clona el repositorio (o descárgalo como ZIP):
   ```bash
   git clone https://github.com/AdidSS/Manos-en-accion.git
   cd Manos-en-accion
   ```
2. Crea un entorno virtual:
    En Windows:

   ```bash
    python -m venv venv
    venv\Scripts\activate
   ```

    En macOS/Linux:
   ```bash
    python3 -m venv venv
    source venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
    pip install -r requirements.txt
   ```

#### Ejecución:
Clasificaciones en tiempo real:

```bash
python real_time_predictions.py
```


