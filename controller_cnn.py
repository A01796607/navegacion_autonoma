"""
Actividad 2.1 — Detección de Señales de Tráfico con CNN
Equipo 20

Luis Alberto Gutiérrez Rivera  A01251467
Pablo Gabriel Galean Benítez   A01735281
Salvador Hernández Medrano      A01796607
Esteban Guerrero Rivero         A01795053

Controlador manual por teclado con detección de señales de tráfico mediante
una CNN basada en arquitectura NVIDIA entrenada sobre el dataset GTSRB (43 clases).

Nota de posicionamiento:
  El vehículo debe colocarse en el CARRIL DERECHO desde el editor de Webots,
  sin posicionarlo sobre la línea amarilla central, para facilitar la visibilidad
  de las señales de tráfico ubicadas a los lados de la carretera.

Teclas de control:
  ↑           — Aumentar velocidad (+2 km/h cada tick)
  ↓           — Reducir velocidad  (-2 km/h cada tick)
  ←           — Girar a la izquierda (ángulo fijo; vuelve a 0 al soltar)
  →           — Girar a la derecha  (ángulo fijo; vuelve a 0 al soltar)
  ESPACIO     — Frenar y centrar dirección

Pipeline de detección:
  1. Captura de imagen desde la cámara frontal del vehículo.
  2. Recorte de tres regiones candidatas (izq / centro / der) en la mitad superior.
  3. Redimensionado a 32×32 px y normalización al rango [0, 1].
  4. Forward pass en numpy con los pesos exportados desde el modelo Keras (joblib).
  5. Si la confianza supera el umbral (60%), se muestra la señal en pantalla.
"""

import numpy as np
import cv2
import onnxruntime as ort
from vehicle import Driver
from controller import Keyboard

LOG_PATH = '/Users/salvadorhernandez/Navegación autonoma/Modulo 4/04_deep_learning/controller_log.txt'
_log_file = open(LOG_PATH, 'w', buffering=1)

def log(msg):
    print(msg, flush=True)
    _log_file.write(msg + '\n')


# ── Parámetros generales ──────────────────────────────────────────────────────

TIME_STEP       = 50        # ms
SPEED_STEP      = 2.0       # km/h por pulsación (↑/↓ acumulan)
MAX_SPEED       = 80.0      # km/h
STEERING_FIXED  = 0.20      # rad fijo mientras ← / → están presionadas
MAX_STEERING    = 0.50      # rad
IMG_SIZE        = 32        # entrada de la CNN
CONF_THRESHOLD  = 0.60      # confianza mínima para reportar señal
DETECT_EVERY    = 3         # clasificar cada N pasos

# ── Ruta al modelo ────────────────────────────────────────────────────────────
MODEL_PATH = '/Users/salvadorhernandez/Navegación autonoma/Modulo 4/04_deep_learning/gtsrb_model.onnx'

# ── Nombres de las 43 clases GTSRB ───────────────────────────────────────────
CLASS_NAMES = [
    'Limite 20km/h',     'Limite 30km/h',     'Limite 50km/h',     'Limite 60km/h',
    'Limite 70km/h',     'Limite 80km/h',     'Fin limite 80',     'Limite 100km/h',
    'Limite 120km/h',    'No rebasar',         'No rebasar >3.5t',  'Prioridad cruce',
    'Via preferente',    'Ceda el paso',       'Alto',              'Sin circulacion',
    'Sin circ. >3.5t',   'Prohibido entrar',   'Precaucion general','Curva pelig. izq',
    'Curva pelig. der',  'Curva doble',        'Camino irregular',  'Camino resbaladizo',
    'Angostamiento der', 'Obras en camino',    'Semaforo',          'Peatones',
    'Zona escolar',      'Ciclistas',          'Hielo/nieve',       'Animales salvajes',
    'Fin restricciones', 'Girar derecha',      'Girar izquierda',   'Seguir adelante',
    'Derecho o derecha', 'Derecho o izquierda','Conservar derecha', 'Conservar izquierda',
    'Glorieta',          'Fin no rebasar',     'Fin no reb. >3.5t'
]




# ── Funciones auxiliares ──────────────────────────────────────────────────────

def get_camera_image(camera):
    width  = camera.getWidth()
    height = camera.getHeight()
    raw    = camera.getImage()
    arr    = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    return arr[:, :, :3].copy()  # BGRA → BGR


def predict_sign(session, image_bgr):
    """
    Evalúa tres recortes de la mitad superior de la imagen y devuelve
    la clase con mayor confianza.
    """
    h, w     = image_bgr.shape[:2]
    inp_name = session.get_inputs()[0].name

    crops = [
        image_bgr[int(0.05*h):int(0.55*h), int(0.05*w):int(0.45*w)],
        image_bgr[int(0.05*h):int(0.55*h), int(0.25*w):int(0.75*w)],
        image_bgr[int(0.05*h):int(0.55*h), int(0.55*w):int(0.95*w)],
    ]

    best_class = 0
    best_conf  = 0.0

    for crop in crops:
        if crop.size == 0:
            continue
        rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
        x     = small[np.newaxis, ...]                          # (1, 32, 32, 3)
        pred  = session.run(None, {inp_name: x})[0][0]
        cid   = int(np.argmax(pred))
        conf  = float(pred[cid])
        if conf > best_conf:
            best_conf  = conf
            best_class = cid

    return best_class, best_conf


def draw_overlay(image_bgr, sign_name, confidence, speed, steering):
    out = image_bgr.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    color = (0, 230, 80) if confidence >= CONF_THRESHOLD else (160, 160, 160)

    cv2.putText(out, f"Senal: {sign_name}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.putText(out, f"Confianza: {confidence*100:.1f}%",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(out, f"Vel: {speed:.0f} km/h   Dir: {steering:+.2f} rad   [flechas]",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 220, 0), 2)

    return out


# ── Bucle principal ───────────────────────────────────────────────────────────

def main():
    driver   = Driver()
    camera   = driver.getDevice("camera")
    camera.enable(TIME_STEP)

    log("=" * 55)
    log("  Actividad 2.1 — Deteccion de Señales  |  Equipo 20")
    log("=" * 55)
    log(f"Cargando modelo desde: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    log("Modelo ONNX cargado.")
    keyboard = driver.getKeyboard()
    keyboard.enable(TIME_STEP)

    log("\nControles (haz clic en la vista 3D de Webots):")
    log("  UP / DOWN  ->  Acelerar / Frenar")
    log("  LEFT / RIGHT  ->  Girar izquierda / derecha")
    log("  ESPACIO ->  Detener\n")

    current_speed    = 0.0
    current_steering = 0.0
    detected_sign    = "Sin señal"
    detected_conf    = 0.0
    frame_count      = 0
    detected_ids     = set()

    driver.setCruisingSpeed(current_speed)
    driver.setSteeringAngle(current_steering)

    cv2.namedWindow("Traffic signs", cv2.WINDOW_AUTOSIZE)

    while driver.step() != -1:

        # ── Teclado Webots (clic en vista 3D para activar) ───────────────────
        key = keyboard.getKey()

        # Dirección: valor fijo mientras se presiona, vuelve a 0 al soltar
        if key == Keyboard.LEFT:
            current_steering = -0.20
        elif key == Keyboard.RIGHT:
            current_steering =  0.20
        elif key not in (Keyboard.UP, Keyboard.DOWN, ord(' ')):
            current_steering = 0.0

        # Velocidad: acumulación lenta
        if key == Keyboard.UP:
            current_speed = min(current_speed + SPEED_STEP, MAX_SPEED)
        elif key == Keyboard.DOWN:
            current_speed = max(current_speed - SPEED_STEP, 0.0)
        elif key == ord(' '):
            current_speed    = 0.0
            current_steering = 0.0

        cv2.waitKey(1)

        driver.setCruisingSpeed(current_speed)
        driver.setSteeringAngle(current_steering)

        # ── Imagen ────────────────────────────────────────────────────────────
        image_bgr = get_camera_image(camera)
        frame_count += 1

        # ── Detección ─────────────────────────────────────────────────────────
        if frame_count % DETECT_EVERY == 0:
            cid, conf = predict_sign(session, image_bgr)

            if conf >= CONF_THRESHOLD:
                detected_sign = CLASS_NAMES[cid]
                detected_conf = conf

                if cid not in detected_ids:
                    detected_ids.add(cid)
                    log(f"[NUEVA] {CLASS_NAMES[cid]:30s}  conf={conf*100:.1f}%  "
                        f"| Detectadas: {len(detected_ids)}/16")
            else:
                detected_sign = "Sin señal"
                detected_conf = conf

        # ── Visualización ─────────────────────────────────────────────────────
        display = cv2.resize(image_bgr, (560, 360))
        display = draw_overlay(
            display, detected_sign, detected_conf,
            current_speed, current_steering
        )
        cv2.imshow("Traffic signs", display)


if __name__ == "__main__":
    main()
