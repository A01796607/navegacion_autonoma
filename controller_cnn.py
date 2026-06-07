"""
Actividad 2.1 — Detección de Señales de Tráfico con CNN
Equipo 20

Luis Alberto Gutiérrez Rivera  A01251467
Pablo Gabriel Galean Benítez   A01735281
Salvador Hernández Medrano      A01796607
Esteban Guerrero Rivero         A01795053

Descripción:
    Controlador manual para Webots que integra detección de señales de tráfico
    en tiempo real. Captura imágenes desde la cámara frontal del vehículo,
    las procesa con una CNN entrenada sobre el dataset GTSRB (43 clases) y
    muestra en pantalla la señal detectada junto con su nivel de confianza,
    velocidad actual y ángulo de dirección. El vehículo se controla con las
    teclas de dirección del teclado a través de la API de Webots.

Modelo:
    CNN inspirada en la arquitectura NVIDIA "End-to-End Learning for
    Self-Driving Cars" (Bojarski et al., 2016), adaptada para clasificación
    multiclase. Entrenada sobre GTSRB con 94.8% de exactitud en prueba.
    Exportada en formato ONNX desde el notebook de entrenamiento.

Nota de posicionamiento:
    El vehículo debe colocarse en el CARRIL DERECHO desde el editor de Webots,
    sin posicionarlo sobre la línea amarilla central, para facilitar la
    visibilidad de las señales de tráfico ubicadas a los lados de la carretera.

Teclas de control:
    ↑       — Aumentar velocidad (+2 km/h por tick, se acumula)
    ↓       — Reducir velocidad  (-2 km/h por tick, se acumula)
    ←       — Girar a la izquierda (ángulo fijo; vuelve a 0 al soltar)
    →       — Girar a la derecha  (ángulo fijo; vuelve a 0 al soltar)
    ESPACIO — Frenar y centrar dirección

Pipeline de detección (por cada DETECT_EVERY ticks):
    1. Captura de imagen BGRA desde la cámara frontal del vehículo.
    2. Extracción de tres recortes de la mitad superior de la imagen
       (izquierda, centro, derecha) para cubrir distintas posiciones de señal.
    3. Conversión a RGB, redimensionado a 32×32 px y normalización a [0, 1].
    4. Inferencia con el modelo ONNX cargado desde gtsrb_model.onnx.
    5. Si la confianza del recorte más seguro supera CONF_THRESHOLD,
       se reporta la señal en pantalla y en el archivo de log.

Log:
    Cada señal detectada por primera vez se registra en controller_log.txt
    con su nombre, nivel de confianza y conteo acumulado de señales únicas.
"""

import numpy as np
import cv2
import onnxruntime as ort
from vehicle import Driver
from controller import Keyboard


# ── Log a archivo (stdout no visible en controlador externo de Webots) ────────

LOG_PATH  = '/Users/salvadorhernandez/Navegación autonoma/Modulo 4/04_deep_learning/controller_log.txt'
_log_file = open(LOG_PATH, 'w', buffering=1)

def log(msg):
    """Imprime en stdout y escribe en el archivo de log simultáneamente."""
    print(msg, flush=True)
    _log_file.write(msg + '\n')


# ── Parámetros generales ──────────────────────────────────────────────────────

TIME_STEP      = 50     # Intervalo de actualización del controlador en ms
SPEED_STEP     = 2.0    # km/h que se suman/restan por tick al mantener ↑/↓
MAX_SPEED      = 80.0   # Velocidad máxima permitida en km/h
STEERING_FIXED = 0.20   # Ángulo de dirección fijo (rad) al presionar ←/→
MAX_STEERING   = 0.50   # Límite máximo de ángulo de dirección en rad
IMG_SIZE       = 32     # Resolución de entrada de la CNN (32×32 px)
CONF_THRESHOLD = 0.80   # Umbral mínimo de confianza para reportar una señal
DETECT_EVERY   = 3      # Ejecutar inferencia cada N ticks para ahorrar cómputo

# ── Ruta al modelo exportado desde el notebook de entrenamiento ───────────────
MODEL_PATH = '/Users/salvadorhernandez/Navegación autonoma/Modulo 4/04_deep_learning/gtsrb_model.onnx'

# ── Etiquetas de las 43 clases del dataset GTSRB ─────────────────────────────
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
    """
    Obtiene la imagen actual de la cámara de Webots como arreglo numpy BGR.

    La API de Webots devuelve los datos en formato BGRA (4 canales).
    Se descartan el canal alfa y se regresa solo BGR para compatibilidad
    con OpenCV.

    Retorna:
        numpy.ndarray de forma (H, W, 3) con dtype uint8.
    """
    width  = camera.getWidth()
    height = camera.getHeight()
    raw    = camera.getImage()
    arr    = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    return arr[:, :, :3].copy()  # BGRA → BGR


def predict_sign(session, image_bgr):
    """
    Clasifica la señal de tráfico más prominente en la imagen.

    Estrategia de recortes múltiples:
        Se evalúan tres regiones de la mitad superior de la imagen
        (izquierda, centro y derecha) para detectar señales ubicadas
        en distintas posiciones del encuadre. Se retorna la clase con
        mayor confianza entre los tres recortes.

    Preprocesamiento por recorte:
        1. Conversión BGR → RGB (el modelo fue entrenado en RGB).
        2. Redimensionado a IMG_SIZE×IMG_SIZE px.
        3. Normalización al rango [0, 1] dividiendo entre 255.
        4. Expansión de dimensión de batch: (32,32,3) → (1,32,32,3).

    Args:
        session:    Sesión ONNX Runtime con el modelo cargado.
        image_bgr:  Imagen capturada de la cámara en formato BGR.

    Retorna:
        (class_id, confidence): índice de clase y confianza [0, 1].
    """
    h, w     = image_bgr.shape[:2]
    inp_name = session.get_inputs()[0].name

    # Tres recortes de la mitad superior: izquierda, centro, derecha
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
        x     = small[np.newaxis, ...]          # añade dimensión de batch
        pred  = session.run(None, {inp_name: x})[0][0]
        cid   = int(np.argmax(pred))
        conf  = float(pred[cid])
        if conf > best_conf:
            best_conf  = conf
            best_class = cid

    return best_class, best_conf


def draw_overlay(image_bgr, sign_name, confidence, speed, steering):
    """
    Dibuja el panel informativo sobre la imagen de la cámara.

    Superpone una banda semitransparente en la parte superior con:
      - Nombre de la señal detectada (verde si confianza >= umbral, gris si no).
      - Porcentaje de confianza de la predicción.
      - Velocidad actual, ángulo de dirección y recordatorio de controles.

    Args:
        image_bgr:  Imagen base en formato BGR.
        sign_name:  Nombre de la clase detectada.
        confidence: Confianza de la predicción [0, 1].
        speed:      Velocidad actual del vehículo en km/h.
        steering:   Ángulo de dirección actual en radianes.

    Retorna:
        Imagen BGR con el overlay aplicado.
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # Banda semitransparente en la parte superior
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # Verde si supera el umbral de confianza, gris si no
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
    """
    Punto de entrada del controlador.

    Inicializa el driver de Webots, carga el modelo ONNX y entra al
    bucle de simulación. En cada tick:
      1. Lee el teclado y actualiza velocidad/dirección del vehículo.
      2. Captura la imagen de la cámara.
      3. Cada DETECT_EVERY ticks ejecuta la inferencia de detección.
      4. Renderiza el panel de visualización con OpenCV.
    """
    driver = Driver()
    camera = driver.getDevice("camera")
    camera.enable(TIME_STEP)

    log("=" * 55)
    log("  Actividad 2.1 — Deteccion de Señales  |  Equipo 20")
    log("=" * 55)
    log(f"Cargando modelo desde: {MODEL_PATH}")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    log("Modelo cargado correctamente.")

    keyboard = driver.getKeyboard()
    keyboard.enable(TIME_STEP)

    log("\nControles (haz clic en la vista 3D de Webots para activar teclado):")
    log("  UP / DOWN    ->  Acelerar / Frenar")
    log("  LEFT / RIGHT ->  Girar izquierda / derecha (se centra al soltar)")
    log("  ESPACIO      ->  Detener completamente\n")

    current_speed    = 0.0
    current_steering = 0.0
    detected_sign    = "Sin señal"
    detected_conf    = 0.0
    frame_count      = 0
    detected_ids     = set()   # clases únicas detectadas durante la sesión

    driver.setCruisingSpeed(current_speed)
    driver.setSteeringAngle(current_steering)

    cv2.namedWindow("Traffic signs", cv2.WINDOW_AUTOSIZE)

    while driver.step() != -1:

        # ── Teclado ───────────────────────────────────────────────────────────
        key = keyboard.getKey()

        # Dirección: ángulo fijo mientras la tecla está presionada;
        # se restablece a 0 automáticamente al soltarla
        if key == Keyboard.LEFT:
            current_steering = -STEERING_FIXED
        elif key == Keyboard.RIGHT:
            current_steering =  STEERING_FIXED
        elif key not in (Keyboard.UP, Keyboard.DOWN, ord(' ')):
            current_steering = 0.0

        # Velocidad: acumulación progresiva con límite superior e inferior
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

        # ── Captura de imagen ─────────────────────────────────────────────────
        image_bgr = get_camera_image(camera)
        frame_count += 1

        # ── Inferencia de detección (cada DETECT_EVERY ticks) ─────────────────
        if frame_count % DETECT_EVERY == 0:
            cid, conf = predict_sign(session, image_bgr)

            if conf >= CONF_THRESHOLD:
                detected_sign = CLASS_NAMES[cid]
                detected_conf = conf

                # Registrar señal nueva (primera vez que se detecta esta clase)
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
