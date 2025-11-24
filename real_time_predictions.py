import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import torch.nn.functional as F
import threading
import queue
import time
import sys


# Configuración
RUTA_MODELO_SEG = 'best_seg.pt'
RUTA_MODELO_CLASIF = 'efficientNet_sign_language_seg.pth'
IMG_SIZE = 224
CONF_THRESHOLD_SEG = 0.5
FPS_TARGET = 30
PREDICTION_INTERVAL = 0.3  # Aumentado para dar más tiempo

CLASS_NAMES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z'
])
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Optimizar memoria GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

# Transformaciones optimizadas
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Cargar modelos con verificación
print("Cargando modelo de segmentación...")
try:
    model_seg = YOLO(RUTA_MODELO_SEG)
    print("Modelo de Segmentación (YOLO) cargado exitosamente.")
    print(f"Clases del modelo YOLO: {model_seg.names}")
except Exception as e:
    print(f"Error cargando modelo de segmentación: {e}")
    sys.exit(1)

print("Cargando modelo de clasificación")
try:
    model_clasif = models.efficientnet_b0(weights=None)
    in_features = model_clasif.classifier[1].in_features
    model_clasif.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model_clasif.load_state_dict(torch.load(RUTA_MODELO_CLASIF, map_location=device))
    model_clasif.to(device)
    model_clasif.eval()
    
    print("Modelo de Clasificación (EfficientNetB0) cargado exitosamente.")
except Exception as e:
    print(f"Error cargando modelo de clasificación: {e}")
    sys.exit(1)

label_map_inverso = {idx: class_name for idx, class_name in enumerate(CLASS_NAMES)}

class RealtimePredictor:
    def __init__(self):
        self.prediction_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.last_prediction_time = 0
        self.current_prediction = ("N/A", 0.0)
        self.bbox = None
        self.running = True
        
    def predict_async(self, frame):
        """Predicción asíncrona en hilo separado"""
        if not self.prediction_queue.full():
            try:
                self.prediction_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
    
    def prediction_worker(self):
        """Worker thread para procesamiento de predicciones"""
        while self.running:
            try:
                frame = self.prediction_queue.get(timeout=1)
                    
                letra, confianza, bbox = self.process_frame_optimized(frame)
                
                try:
                    self.result_queue.put_nowait((letra, confianza, bbox))
                except queue.Full:
                    # Si la cola está llena, descarta el resultado más antiguo
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((letra, confianza, bbox))
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en prediction worker: {e}")
                import traceback
                traceback.print_exc()
    
    def process_frame_optimized(self, frame):
        """Procesamiento optimizado de frame único"""
        try:
            
            # Segmentación YOLO
            with torch.no_grad():
                resultados_seg = model_seg.predict(
                    frame, 
                    conf=CONF_THRESHOLD_SEG, 
                    verbose=False,
                    imgsz=416,
                    classes=[0],  # Solo clase 0 (person/hand)
                    save=False
                )

            # Verificar si hay máscaras
            if resultados_seg[0].masks is None or len(resultados_seg[0].masks) == 0:
                return "N/A", 0.0, None

            # Verificar si hay bounding boxes
            if resultados_seg[0].boxes is None or len(resultados_seg[0].boxes) == 0:
                return "N/A", 0.0, None

            # Obtener bounding box
            bbox = resultados_seg[0].boxes.xyxy[0].cpu().numpy().astype(int)
            
            # Encontrar máscara principal
            mascara_principal = resultados_seg[0].masks[0]  # Tomar la primera por simplicidad
            
            # Crear máscara binaria
            mascara_binaria = np.zeros(frame.shape[:2], dtype=np.uint8)
            contorno_principal = mascara_principal.xy[0].astype(np.int32)
            cv2.fillPoly(mascara_binaria, [contorno_principal], 255)

            # Verificar que la máscara no esté vacía
            if np.sum(mascara_binaria) == 0:
                return "N/A", 0.0, None

            # Aplicar máscara
            img_fondo_negro = cv2.bitwise_and(frame, frame, mask=mascara_binaria)

            # Verificar que la imagen resultante no esté completamente negra
            if np.sum(img_fondo_negro) == 0:
                return "N/A", 0.0, None

            # Optimizar conversión y transformación
            img_pil = Image.fromarray(cv2.cvtColor(img_fondo_negro, cv2.COLOR_BGR2RGB))
            input_tensor = val_transform(img_pil).unsqueeze(0).to(device)

            # Clasificación
            with torch.no_grad():
                logits = model_clasif(input_tensor)
                probabilidades = F.softmax(logits, dim=1)
                confianza, pred_idx = torch.max(probabilidades, dim=1)
                letra_predicha = label_map_inverso[pred_idx.item()]
                confianza_predicha = confianza.item()
                return letra_predicha, confianza_predicha, bbox

        except Exception as e:
            print(f"Error en process_frame_optimized: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0, None
    
    def get_latest_result(self):
        """Obtener el resultado más reciente sin bloquear"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.current_prediction = result[:2]
                self.bbox = result[2]
        except queue.Empty:
            pass
        return self.current_prediction, self.bbox
    
    def stop(self):
        """Detener el predictor"""
        self.running = False

def mostrar_camara():
    """Función principal para tiempo real"""
    predictor = RealtimePredictor()
    
    # Iniciar worker thread
    prediction_thread = threading.Thread(target=predictor.prediction_worker, daemon=True)
    prediction_thread.start()
    
    # Configurar cámara
    print("Iniciando cámara")
    cam = cv2.VideoCapture(0) # 0 es la de la compu
                              # 1 es la de la webcam
    
    if not cam.isOpened():
        print("Error: No se pudo abrir la cámara")
        sys.exit(1)

    # Optimizar configuración de cámara
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cam.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Variables para control de timing
    last_prediction_time = 0
    frame_count = 0

    try:
        print("Iniciando detección en tiempo real (presiona 'q' para salir)...")
        
        while True:
            ret, frame = cam.read()
            if not ret:
                print("No se pudo leer frame de la cámara")
                continue
            
            current_time = time.time()
            frame_count += 1
            
            # Predecir solo cada PREDICTION_INTERVAL segundos
            if current_time - last_prediction_time >= PREDICTION_INTERVAL:
                predictor.predict_async(frame)
                last_prediction_time = current_time
            
            # Obtener resultado más reciente
            (letra_predicha, confianza_predicha), bbox = predictor.get_latest_result()
            
            # Aplicar filtro de confianza MÁS BAJO
            umbral_confianza = 0.5
            if confianza_predicha < umbral_confianza:
                letra_predicha = "N/A"
            
            # Dibujar interfaz
            display_frame = frame.copy()
            
            # Información principal
            cv2.putText(display_frame, "Manos en Accion", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 86, 255), 3)
            
            # Dibujar bounding box si hay detección
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (56, 112, 71), 2)
                
                # Mostrar predicción
                cv2.putText(display_frame, f"{letra_predicha} ({confianza_predicha:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 63), 2)
            
            # Instrucciones
            cv2.putText(display_frame, "q: salir", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 155, 20), 2)
            
            cv2.imshow("Manos en Accion: Deteccion de LSM", display_frame)
            
            # Control de salida
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDetección interrumpida por el usuario")

    finally:
        print("\nLimpiando recursos...")
        predictor.stop()
        cam.release()
        cv2.destroyAllWindows()
        print("Detección finalizada")

# Llamar función
if __name__ == "__main__":
    mostrar_camara()