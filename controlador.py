"""

Actividad 3.1 - Detección de Peatones con SVM  - Equipo 20

Luis Alberto Gutiérrez Rivera A01251467
Pablo Gabriel Galean Benítez A01735281
Salvador Hernández Medrano A01796607
Esteban Guerrero Rivero A01795053

"""


# 'vehicle' es un módulo interno de Webots — solo existe cuando el script
# corre dentro del simulador. No puede importarse desde la terminal directamente.
from vehicle import Driver
import cv2
import numpy as np
import math
import os
import logging
import joblib
from skimage.feature import hog

# Log se limpia en cada ejecución (filemode='w')
LOG_PATH = os.path.join(os.path.dirname(__file__), "controlador.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# =========================
# Parámetros generales
# =========================

TIME_STEP = 5               #tiempo de actualización del controlador en ms
TARGET_SPEED_KMH = 50.0     #km/hr
MAX_STEERING_ANGLE = 0.40   #input dirección en rads

# Ganancias PID.
KP = 0.006                  #ganancia proporcional (intensidad de la respuesta al error actual)
KI = 0.0                    #ganancia integral (acumula errores en el tiempo) 
KD = 0.0001                 #ganancia derivativa (mide la tasa de cambio del error) 

# Canny
CANNY_LOW = 40              #umbral inferior (bordes debiles entre low y high)
CANNY_HIGH = 120            #umbral superior (bordes fuertes arriba)
                            #parametros bajos detectan mucho ruido
                            #parametros altos dejan de captar lineas

# HoughLinesP
RHO = 2                     #resolución en px para la distancia p de Hough 
THETA = np.pi / 180         #resolución angular 1° equiv
THRESHOLD = 10              #cantidad minima para aceptar linea (bajo mas ruido, alto puede perder curvas)
MIN_LINE_LENGTH = 8         #longitud mínima en px para aceptar una línea
MAX_LINE_GAP = 35           #distancia máxima entre segmentos para considerarlos una sola linea

# Filtro de líneas casi horizontales.
MIN_ABS_SLOPE = 0.10        #ignora línea casi horizontales

# =========================
# Parámetros detección de peatones (SVM + HOG)
# =========================

# Ruta al modelo exportado desde el notebook (un nivel arriba del controlador)
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "svm_inria_hog.joblib")

# Tamaño de ventana igual al de entrenamiento (Dalal & Triggs)
SW_WIN_W      = 64
SW_WIN_H      = 128

# Escalas a buscar: 1.0 → 64×128, 0.75 → 48×96, 0.5 → 32×64
SW_SCALES     = [0.5]        # única escala útil con imagen 256×128
SW_STEP_X     = 32          # paso horizontal en px (escala 1.0)
SW_STEP_Y     = 32          # paso vertical en px (escala 1.0)

# ROI para sliding window: escanear el 90 % superior de la imagen
SW_ROI_RATIO  = 0.90        # peatones aparecen en zona media-baja del encuadre

# =========================
# Parámetros LiDAR (Sick LMS 291)
# =========================

LIDAR_DEVICE      = "Sick LMS 291"
LIDAR_ANGLE_DEG   = 20          # ángulo total de lectura del LiDAR (20° frontal)
LIDAR_HALF_AREA   = 20          # rayos a cada lado del centro (≈20° con el Sick LMS 291)
OBSTACLE_MAX_DIST = 20.0        # metros — límite de detección (rubrica)
OBSTACLE_BRAKE_DIST = 12.0      # metros — umbral real de frenado (evita falsos positivos de paredes lejanas)
                                # se usa un umbral menor a MAX_DIST porque paredes laterales lejanas
                                # pueden caer dentro del cono frontal y generar detecciones falsas

# Estados del vehículo
STATE_NORMAL      = "LIBRE"
STATE_BARREL      = "BARRIL"
STATE_PEDESTRIAN  = "PEATON"


# Setpoint = centro horizontal de la imagen (width/2).
# Variable de proceso = posición X de la línea de carril extrapolada a y=85% de la imagen.
# El PID calcula cuánto girar el volante para mantener la línea centrada en cuadro.
class PIDController:
    """Controlador PID discreto para el ángulo de dirección del vehículo."""

    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)):
        """
        Inicializa el controlador PID.

        Args:
            Kp: Ganancia proporcional.
            Ki: Ganancia integral.
            Kd: Ganancia derivativa.
            setpoint: Valor deseado de la variable de proceso.
            output_limits: Tupla (min, max) para saturar la salida.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0
        self.min_output, self.max_output = output_limits

    def reset(self):
        """Reinicia el término integral y el error previo (útil al cambiar de estado)."""
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, process_variable, dt):
        """
        Calcula la salida del controlador para un paso de tiempo.

        Args:
            process_variable: Valor actual de la variable controlada (posición X de la línea).
            dt: Delta de tiempo en segundos desde el paso anterior.

        Returns:
            Tupla (output, error): salida saturada del PID y error actual.
        """
        if dt <= 0:
            dt = 1e-6

        # Cálculo del error
        error = self.setpoint - process_variable

        # Proporcional
        P_out = self.Kp * error

        # Integral 
        self.integral += error * dt
        I_out = self.Ki * self.integral

        # Derivative 
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative

        # Producto suma
        output = P_out + I_out + D_out

        # Actualización del error
        self.previous_error = error

        # Saturación por seguridad 
        output = max(self.min_output, min(self.max_output, output))

        return output, error


def get_camera_image(camera):
    """
    Captura el frame actual de la cámara Webots y lo convierte a formato BGR.

    Args:
        camera: Dispositivo de cámara habilitado de Webots.

    Returns:
        np.ndarray de forma (height, width, 3) en formato BGR, tipo uint8.
        El canal Alpha original de Webots (BGRA) es descartado.
    """
    width = camera.getWidth()
    height = camera.getHeight()

    raw_image = camera.getImage()
    image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
    image_bgr = image_array[:, :, :3].copy()

    return image_bgr

def region_of_interest(edge_image):
    """
    Aplica una máscara trapezoidal para aislar la zona del carril en la imagen de bordes.

    El trapecio cubre desde la base completa hasta el 65% de la altura de la imagen,
    descartando el horizonte y los márgenes laterales donde no hay carril relevante.

    Args:
        edge_image: Imagen binaria de bordes (salida de Canny), forma (height, width).

    Returns:
        Tupla (roi_image, vertices): imagen recortada con la máscara aplicada
        y array de vértices del trapecio usado.
    """
    height, width = edge_image.shape

    vertices = np.array([[
        (int(0.00 * width), height),
        (int(0.25 * width), int(0.65 * height)),
        (int(0.75 * width), int(0.65 * height)),
        (int(1.00 * width), height)
    ]], dtype=np.int32)

    mask = np.zeros_like(edge_image)
    cv2.fillPoly(mask, vertices, 255)

    roi_image = cv2.bitwise_and(edge_image, mask)

    return roi_image, vertices


def detect_lines(image_bgr):
    """
    Pipeline completo de detección de líneas de carril sobre una imagen BGR.

    Ejecuta: Grayscale → GaussianBlur → Canny → ROI → HoughLinesP.

    Args:
        image_bgr: Imagen de cámara en formato BGR, forma (height, width, 3).

    Returns:
        Tupla (lines, debug):
            - lines: Lista de segmentos [[x1, y1, x2, y2], ...] o None si no hay líneas.
            - debug: Diccionario con imágenes intermedias ('gray', 'blur', 'edges',
              'roi_edges', 'roi_vertices') y 'num_hough_lines'.
    """
    # Conversión a escala de grises.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Suavizado para reducir ruido antes de Canny.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detección de bordes con Canny.
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # Aplicación de ROI.
    roi_edges, roi_vertices = region_of_interest(edges)

    # Detección de líneas rectas mediante Transformada de Hough 
    lines = cv2.HoughLinesP(
        roi_edges,
        RHO,
        THETA,
        THRESHOLD,
        np.array([]),
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    debug = {
        "gray": gray,
        "blur": blur,
        "edges": edges,
        "roi_edges": roi_edges,
        "roi_vertices": roi_vertices,
        "num_hough_lines": 0 if lines is None else len(lines)
    }

    return lines, debug


def line_midpoint_x(line):
    """Retorna la coordenada X del punto medio de un segmento (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2.0


def line_slope(line):
    """
    Calcula la pendiente de un segmento (x1, y1, x2, y2).

    Returns:
        Pendiente como float. Retorna math.inf si el segmento es vertical (dx == 0).
    """
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:
        return math.inf

    return dy / dx


def compute_best_process_variable(lines, setpoint, image_height):
    """
    Selecciona la línea de carril más relevante y calcula la variable de proceso para el PID.

    Filtra líneas casi horizontales (|pendiente| < MIN_ABS_SLOPE) y elige la línea
    que minimiza un score combinado: penaliza distancia al setpoint y premia
    líneas cercanas a la base de la imagen (más estables).

    Args:
        lines: Lista de segmentos de HoughLinesP o None.
        setpoint: Centro horizontal de la imagen en píxeles.
        image_height: Alto de la imagen en píxeles.

    Returns:
        Tupla (process_variable, best_line, valid_lines_count):
            - process_variable: Coordenada X de la línea ganadora extrapolada a y=85%,
              o None si no hay líneas válidas.
            - best_line: Tupla (x1, y1, x2, y2) de la línea seleccionada, o None.
            - valid_lines_count: Número de líneas que pasaron el filtro de pendiente.
    """
    if lines is None:
        return None, None, 0

    best_process_variable = None
    best_line = None
    best_score = None
    valid_lines = 0

    for line_group in lines:
        x1, y1, x2, y2 = line_group[0]
        line = (x1, y1, x2, y2)

        slope = line_slope(line)

        if abs(slope) < MIN_ABS_SLOPE:
            continue

        valid_lines += 1

        # Se calcula la posición x de la línea en la parte baja de la imagen
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            y_target = int(image_height * 0.85)
            x_at_bottom = (y_target - b) / m
        else:
            x_at_bottom = x1

        midpoint_y = (y1 + y2) / 2.0
        error = setpoint - x_at_bottom

        # Menor score = mejor línea. abs(error) penaliza líneas alejadas del centro;
        # restar midpoint_y premia las líneas más cercanas a la base (más estables y confiables).
        score = abs(error) - 0.8 * midpoint_y

        if best_score is None or score < best_score:
            best_score = score
            best_process_variable = x_at_bottom
            best_line = line

    return best_process_variable, best_line, valid_lines


def sliding_window_pedestrian(image_bgr, svm_model):
    """
    Sliding Window Search sobre la imagen de cámara (256×128).
    Aplica ROI para ignorar la zona de carretera (parte inferior).
    Retorna lista de bounding boxes (x1, y1, x2, y2) con peatones detectados.
    """
    # Esta función es costosa (~200ms con HOG en Python puro).
    # Solo se llama desde el loop principal cada 60 pasos para no bloquear la simulación.
    detections = []
    img_h = image_bgr.shape[0]

    # ROI: solo el 70 % superior de la imagen
    roi_h = int(img_h * SW_ROI_RATIO)
    roi = image_bgr[0:roi_h, :]

    for scale in SW_SCALES:
        win_w = int(SW_WIN_W * scale)
        win_h = int(SW_WIN_H * scale)

        if win_w > roi.shape[1] or win_h > roi.shape[0]:
            continue

        step_x = max(1, int(SW_STEP_X * scale))
        step_y = max(1, int(SW_STEP_Y * scale))

        for y in range(0, roi.shape[0] - win_h + 1, step_y):
            for x in range(0, roi.shape[1] - win_w + 1, step_x):
                patch = roi[y:y + win_h, x:x + win_w]
                patch_resized = cv2.resize(patch, (SW_WIN_W, SW_WIN_H))
                gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
                feat = hog(gray,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           transform_sqrt=False,
                           feature_vector=True)
                if svm_model.predict([feat])[0] == 1:
                    detections.append((x, y, x + win_w, y + win_h))

    return detections


def process_sick_data(range_image, sick_width, sick_fov):
    """
    Revisa los rayos dentro del ángulo frontal (LIDAR_HALF_AREA a cada lado del centro).
    Retorna (obstacle_angle_rad, obstacle_dist_m) o (None, None) si no hay obstáculo.
    """
    if not range_image or sick_width == 0:
        return None, None

    sumx = 0
    collision_count = 0
    obstacle_dist = 0.0

    for x in range(sick_width // 2 - LIDAR_HALF_AREA, sick_width // 2 + LIDAR_HALF_AREA):
        dist = range_image[x]
        if dist < OBSTACLE_MAX_DIST:
            sumx += x
            collision_count += 1
            obstacle_dist += dist

    if collision_count == 0:
        return None, None

    obstacle_dist /= collision_count
    # Normaliza la posición promedio del obstáculo en [0,1], centra en 0 con -0.5,
    # y escala al FOV real del sensor para obtener el ángulo en radianes.
    obstacle_angle = (sumx / collision_count / sick_width - 0.5) * sick_fov

    return obstacle_angle, obstacle_dist


def classify_obstacle(obstacle_dist, pedestrian_detections):
    """
    - SVM detecta peatón → STATE_PEDESTRIAN (independiente del LiDAR)
    - LiDAR detecta obstáculo sin peatón confirmado → STATE_BARREL
    - Sin detección → STATE_NORMAL
    """
    # PEDESTRIAN tiene prioridad sobre BARREL: si ambos sensores detectan algo
    # simultáneamente, se trata siempre como peatón para maximizar la seguridad.
    if pedestrian_detections:
        return STATE_PEDESTRIAN
    if obstacle_dist is not None and obstacle_dist < OBSTACLE_BRAKE_DIST:
        return STATE_BARREL
    return STATE_NORMAL


def draw_debug_image(image_bgr, lines, best_line, setpoint, roi_vertices,
                     obstacle_dist=None, obstacle_angle=None,
                     pedestrian_detections=None, vehicle_state=STATE_NORMAL):
    """
    - ROI en amarillo.
    - Setpoint en verde.
    - Líneas Hough en azul.
    - Línea seleccionada en rojo.

    """
    debug_image = image_bgr.copy()
    height, width = debug_image.shape[:2]

    cv2.polylines(debug_image, roi_vertices, isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.line(debug_image, (int(setpoint), 0), (int(setpoint), height), (0, 255, 0), 2)

    if lines is not None:
        for line_group in lines:
            x1, y1, x2, y2 = line_group[0]
            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if best_line is not None:
        x1, y1, x2, y2 = best_line
        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 4)

        midpoint_x = int(line_midpoint_x(best_line))
        midpoint_y = int((y1 + y2) / 2)
        cv2.circle(debug_image, (midpoint_x, midpoint_y), 5, (0, 0, 255), -1)

    # Overlay LiDAR: distancia y ángulo del obstáculo detectado
    if obstacle_dist is not None:
        color = (0, 0, 255)
        label = f"OBSTACULO  {obstacle_dist:.1f} m  {math.degrees(obstacle_angle):.1f} deg"
        cv2.putText(debug_image, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(debug_image, "LiDAR: libre", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay SVM: bounding boxes de peatones detectados (magenta)
    if pedestrian_detections:
        for (x1, y1, x2, y2) in pedestrian_detections:
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(debug_image, f"PEATON ({len(pedestrian_detections)})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Overlay estado del vehículo y luces intermitentes
    state_colors = {
        STATE_NORMAL:     (0, 255, 0),
        STATE_PEDESTRIAN: (255, 0, 255),
        STATE_BARREL:     (0, 0, 255),
    }
    state_color = state_colors.get(vehicle_state, (255, 255, 255))
    cv2.putText(debug_image, f"ESTADO: {vehicle_state}", (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # setHazardFlashers() real no se llama: el mundo .wbt tiene un archivo de audio inválido
    # ('c ') en el engine_speaker del BMW X5 que crashea Webots al activar el sistema de luces.
    # Se muestra solo como texto en el overlay como indicador visual equivalente.
    flashers_on = vehicle_state == STATE_BARREL
    flasher_label = "INTERMITENTES: ON" if flashers_on else "INTERMITENTES: OFF"
    flasher_color = (0, 215, 255) if flashers_on else (180, 180, 180)
    cv2.putText(debug_image, flasher_label, (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, flasher_color, 2)

    return debug_image


def main():
    """
    Punto de entrada del controlador. Inicializa todos los dispositivos y ejecuta
    el loop principal de simulación hasta que Webots detiene el controlador.

    Dispositivos inicializados:
        - Cámara (256×128): seguimiento de carril y detección de peatones (SVM).
        - LiDAR Sick LMS 291: detección de obstáculos frontales.

    Loop principal (cada TIME_STEP ms):
        1. Captura imagen → detección de líneas → PID → ángulo de dirección.
        2. Lectura LiDAR → clasificación de obstáculo.
        3. Cada 60 pasos: sliding window SVM para peatones.
        4. Aplica velocidad/freno según estado (LIBRE / BARRIL / PEATON).
        5. Actualiza ventanas de debug OpenCV cada 10 pasos.
    """
    driver = Driver()

    camera = driver.getDevice("camera")
    camera.enable(TIME_STEP)


    lidar = driver.getDevice(LIDAR_DEVICE)
    if lidar is None:
        print(f"ADVERTENCIA: no se encontró el dispositivo '{LIDAR_DEVICE}'. LiDAR desactivado.")
    else:
        lidar.enable(TIME_STEP)
    # getHorizontalResolution() — correcto para LiDAR 2D (escáner de rango).
    # getNumberOfPoints() es para LiDARs 3D con point cloud habilitado y lanza error aquí.
    sick_width = lidar.getHorizontalResolution() if lidar else 0
    sick_fov   = lidar.getFov()                  if lidar else 0.0

    # Carga del modelo SVM para detección de peatones
    try:
        svm_model = joblib.load(MODEL_PATH)
        log.info(f"Modelo SVM cargado: {MODEL_PATH}")
    except Exception as e:
        svm_model = None
        log.warning(f"No se pudo cargar el modelo SVM — {e}")

    pid = None

    driver.setCruisingSpeed(TARGET_SPEED_KMH)
    driver.setSteeringAngle(0.0)

    log.info("Controlador iniciado.")
    log.info(f"Velocidad objetivo: {TARGET_SPEED_KMH} km/h")
    log.info(f"Ganancias PID: Kp={KP} Ki={KI} Kd={KD}")
    log.info(f"LiDAR: {LIDAR_DEVICE}  sick_width={sick_width}  fov={math.degrees(sick_fov):.1f}°")
  

    previous_time = driver.getTime()

    # Ventanas de OpenCV redimensionables.
    cv2.namedWindow("Camera + Hough + PID debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera + Hough + PID debug", 800, 600)

    cv2.namedWindow("ROI Canny", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Canny", 400, 300)

    last_steering_angle = 0.0
    step_counter = 0
    pedestrian_detections = []      # resultado del último SVM (se reutiliza entre pasos)
    previous_state = STATE_NORMAL   # para detectar cambios de estado

    while driver.step() != -1:
        step_counter += 1

        current_time = driver.getTime()
        dt = current_time - previous_time
        previous_time = current_time

        image_bgr = get_camera_image(camera)
        height, width = image_bgr.shape[:2]

        # Setpoint solicitado: valor medio del ancho de la imagen.
        setpoint = width / 2.0

        # Inicializa el PID cuando ya se conoce el ancho real de la imagen.
        if pid is None:
            pid = PIDController(KP, KI, KD, setpoint)

        pid.setpoint = setpoint

        lines, debug = detect_lines(image_bgr)

        # Variable de proceso: punto medio horizontal de la línea detectada.
        process_variable, best_line, _ = compute_best_process_variable(lines, setpoint, height)

        if process_variable is None:
            steering_angle = 0.0
        else:
            pid_output, _ = pid.compute(process_variable, dt)
            # Negativo porque el PID corrige hacia la derecha cuando la línea está a la izquierda
            # del setpoint, pero Webots espera ángulo positivo = giro a la derecha del vehículo.
            steering_angle = -pid_output
            last_steering_angle = steering_angle

        # LiDAR — detección de obstáculos frontales
        range_image = lidar.getRangeImage() if lidar else None
        obstacle_angle, obstacle_dist = process_sick_data(range_image, sick_width, sick_fov)

        # SVM — detección de peatones cada 60 pasos (~300 ms).
        # El resultado se reutiliza entre ejecuciones: pedestrian_detections persiste
        # hasta el siguiente ciclo de 60 pasos para mantener el estado activo.
        if svm_model is not None and step_counter % 60 == 0:
            pedestrian_detections = sliding_window_pedestrian(image_bgr, svm_model)
            if pedestrian_detections:
                log.info(f"[{step_counter}] SVM: {len(pedestrian_detections)} peatón(es)")

        # Clasificación del obstáculo y control de velocidad + luces
        vehicle_state = classify_obstacle(obstacle_dist, pedestrian_detections)

        if vehicle_state in (STATE_BARREL, STATE_PEDESTRIAN):
            # setCruisingSpeed(0) desactiva el crucero pero el auto se desliza por inercia.
            # setBrakeIntensity(1.0) aplica freno físico máximo para detención real.
            # Al reanudar, liberar el freno antes de reactivar la velocidad de crucero.
            driver.setCruisingSpeed(0.0)
            driver.setBrakeIntensity(1.0)
        else:
            driver.setBrakeIntensity(0.0)
            driver.setCruisingSpeed(TARGET_SPEED_KMH)

        # Loguear solo al cambiar de estado
        if vehicle_state != previous_state:
            log.warning(
                f"[{step_counter}] ESTADO: {previous_state} → {vehicle_state}"
            )
            previous_state = vehicle_state

        driver.setSteeringAngle(steering_angle)

        debug_image = draw_debug_image(
            image_bgr,
            lines,
            best_line,
            setpoint,
            debug["roi_vertices"],
            obstacle_dist,
            obstacle_angle,
            pedestrian_detections,
            vehicle_state
        )

        # Actualizar ventanas cada 10 pasos (~50 ms) para reducir carga de display
        if step_counter % 10 == 0:
            cv2.imshow("Camera + Hough + PID debug", debug_image)
            cv2.imshow("ROI Canny", debug["roi_edges"])
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
