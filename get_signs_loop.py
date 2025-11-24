import cv2
import os
import time
import sys

# Configuración
IMG_FOLDERS = "IMGS"
CLASS_NAMES = sorted([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U',
    'V', 'W', 'X', 'Y', 'Z'
])
numero_de_imagenes_por_clase = 10
tiempo_espera = 2  # 2 segundos para cambiar posición
num_corridas = 14 #Ir actualizando para que no se pierdan las imágenes anteriores

#10 y 14 para las imagenes de train IMG_FOLDERS = "IMGS"

# Crear estructura de carpetas
os.makedirs(IMG_FOLDERS, exist_ok=True)
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(IMG_FOLDERS, class_name), exist_ok=True)

# Inicializar cámara
print("Iniciando cámara...")
cam = cv2.VideoCapture(1) # 0 compu, 1 webcam

if not cam.isOpened():
    print("Error: No se pudo abrir la cámara")
    sys.exit(1)

# Configurar resolución
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Cámara inicializada correctamente")
print(f"Las imágenes se guardarán en: {IMG_FOLDERS}")
print(f"Se capturarán {numero_de_imagenes_por_clase} imágenes por cada letra")
print("\nInstrucciones:")
print("- Presiona 'c' para capturar una imagen")
print("- Presiona 's' para saltar a la siguiente letra")
print("- Presiona 'q' para salir")
print("\n" + "="*50)

try:
    while True:
        num_archivo = numero_de_imagenes_por_clase*num_corridas 
        for class_idx, class_name in enumerate(CLASS_NAMES):
            print(f"\nCAPTURANDO LETRA: {class_name} ({class_idx + 1}/{len(CLASS_NAMES)})")
            print(f"Imágenes necesarias: {numero_de_imagenes_por_clase}")
        
            # Cuenta regresiva inicial
            for countdown in range(tiempo_espera, 0, -1):
                ret, frame = cam.read()
                if not ret:
                    print("Error al leer frame de la cámara")
                    continue
                    
                # Añadir texto a la imagen
                display_frame = frame.copy()
                
                # Información principal
                cv2.putText(display_frame, f"LETRA: {class_name}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Preparate... {countdown}s", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Clase {class_idx + 1}/{len(CLASS_NAMES)}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, "c:capturar | s:siguiente | q:salir", 
                        (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Captura de Lenguaje de Señas", display_frame)
                
                key = cv2.waitKey(1000) & 0xFF  # Espera 1 segundo
                if key == ord('q'):
                    raise KeyboardInterrupt
                elif key == ord('s'):
                    break
            
            # Captura de imágenes para la clase actual
            images_captured = 0
            
            while images_captured < numero_de_imagenes_por_clase:
                ret, frame = cam.read()
                if not ret:
                    print(f"Error al capturar imagen {images_captured + 1}")
                    continue
                
                # Preparar frame para mostrar
                display_frame = frame.copy()
                
                # Información de captura
                cv2.putText(display_frame, f"LETRA: {class_name}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Imagen {images_captured + 1}/{numero_de_imagenes_por_clase}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_frame, "Presiona 'c' para capturar", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "c:capturar | s:siguiente | q:salir", 
                        (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Mostrar preview
                cv2.imshow("Captura de Lenguaje de Señas", display_frame)
                
                # Esperar input del usuario
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    # Capturar imagen
                    img_name = f"{class_name}_{images_captured+ num_archivo + 1}.png"
                    img_path = os.path.join(IMG_FOLDERS, class_name, img_name)
                    
                    if cv2.imwrite(img_path, frame):
                        images_captured += 1
                        print(f"Imagen {images_captured}/{numero_de_imagenes_por_clase} guardada: {img_path}")
                        
                        # Mostrar confirmación visual
                        confirm_frame = frame.copy()
                        cv2.putText(confirm_frame, "CAPTURADA", 
                                (confirm_frame.shape[1]//2 - 100, confirm_frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.imshow("Captura de Lenguaje de Señas", confirm_frame)
                        cv2.waitKey(500)  # Mostrar confirmación por 0.5 segundos
                    else:
                        print(f"Error al guardar imagen: {img_path}")
                        
                elif key == ord('s'):
                    print(f"Saltando a la siguiente letra...")
                    break
                elif key == ord('q'):
                    raise KeyboardInterrupt
        num_corridas = num_corridas + 1
        if key ==ord('q'):
            break

except KeyboardInterrupt:
    print("\n\nCaptura interrumpida por el usuario")

finally:
    # Limpieza de recursos
    print("\nLimpiando recursos...")
    cam.release()
    cv2.destroyAllWindows()
    print("Captura finalizada")