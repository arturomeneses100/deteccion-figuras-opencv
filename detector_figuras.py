#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 DETECTOR DE FIGURAS GEOMÉTRICAS POR COLOR (ROJO, VERDE Y AZUL) EN TIEMPO REAL
==============================================================================
 Curso de Programación en Python - Visión artificial con OpenCV

 Flujo de procesamiento (para cada cuadro de la cámara):

   1. Captura          -> se lee un cuadro (frame) de la webcam.
   2. Preprocesamiento -> suavizado (filtro Gaussiano) para eliminar ruido.
   3. Conversión HSV   -> se pasa de BGR a HSV (tono, saturación, brillo).
   4. Umbralización    -> una máscara binaria por color (rojo, verde, azul).
   5. Morfología       -> apertura (quita puntos) y cierre (rellena huecos).
   6. Contornos        -> se buscan las siluetas blancas de cada máscara.
   7. Validación       -> área, solidez, pureza de color, bordes, convexidad.
   8. Clasificación    -> triángulo, cuadrado, rectángulo o círculo.
                          Si la forma es dudosa, SE DESCARTA.
   9. Estabilidad      -> una figura sólo se muestra si aparece en varios
                          cuadros seguidos (evita falsos positivos fugaces).
  10. Visualización    -> se dibujan sólo las figuras confirmadas.

 Teclas:
   q / ESC : salir
   m       : mostrar / ocultar las máscaras de cada color
   c       : cambiar el color cuyos rangos HSV se editan (también con barra)
   s       : guardar la configuración en config_figuras.json
   r       : recargar la configuración por defecto
   p       : pausar / reanudar

 Uso:
   python detector_figuras.py                 # cámara 0
   python detector_figuras.py --camara 1      # otra cámara
   python detector_figuras.py --imagen foto.jpg   # probar con una imagen
==============================================================================
"""

import argparse
import copy
import json
import math
import os
from collections import Counter, deque

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN POR DEFECTO
# ---------------------------------------------------------------------------
# En OpenCV el canal H (tono) va de 0 a 179 (grados / 2); S y V van de 0 a 255.
# Si h_min > h_max se interpreta que el rango "da la vuelta" por el 0.
# Esto es justo lo que pasa con el ROJO, que está en ambos extremos del círculo
# de tono (≈170-179 y ≈0-10).
CONFIG_POR_DEFECTO = {
    "colores": {
        "rojo":  {"h_min": 170, "h_max": 8,   "s_min": 120, "s_max": 255, "v_min": 70, "v_max": 255},
        "verde": {"h_min": 40,  "h_max": 85,  "s_min": 90,  "s_max": 255, "v_min": 50, "v_max": 255},
        "azul":  {"h_min": 95,  "h_max": 128, "s_min": 120, "s_max": 255, "v_min": 50, "v_max": 255},
    },
    # Filtros de contorno
    "area_min": 1500,          # px² - descarta puntos, reflejos y ruido
    "area_max": 120000,        # px² - descarta regiones enormes (fondo, pared)
    "solidez_min": 0.90,       # área / área de la envolvente convexa (0-1)
    "pureza_min": 0.85,        # fracción de píxeles del color dentro del contorno
    "margen_borde": 5,         # px - descarta figuras cortadas por el borde
    # Morfología
    "kernel": 5,               # tamaño del elemento estructurante (impar)
    "iter_apertura": 2,
    "iter_cierre": 2,
    "blur": 7,                 # tamaño del filtro Gaussiano (impar)
    # Aproximación poligonal y clasificación
    "epsilon": 0.03,           # tolerancia de approxPolyDP (fracción del perímetro)
    "circularidad_min": 0.82,  # 4πA/P² ; 1.0 = círculo perfecto
    "ajuste_min": 0.80,        # qué tanto llena la figura su forma ideal
    "tolerancia_cuadrado": 0.12,  # |1 - lado_corto/lado_largo| para ser cuadrado
    # Estabilidad temporal
    "cuadros_confirmar": 5,    # cuadros seguidos para mostrar una figura
    "cuadros_olvidar": 3,      # cuadros sin verla para eliminarla
    # Imagen
    "ancho_proceso": 640,      # se redimensiona el cuadro a este ancho
}

ARCHIVO_CONFIG = "config_figuras.json"

# Color (BGR) con que se dibuja cada figura en pantalla
COLOR_DIBUJO = {"rojo": (0, 0, 255), "verde": (0, 200, 0), "azul": (255, 80, 0)}
NOMBRES_COLOR = ["rojo", "verde", "azul"]


def cargar_config(ruta=ARCHIVO_CONFIG):
    """Lee la configuración del archivo JSON; si no existe, usa la de fábrica."""
    cfg = copy.deepcopy(CONFIG_POR_DEFECTO)
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                guardada = json.load(f)
            cfg.update({k: v for k, v in guardada.items() if k != "colores"})
            for nombre, rango in guardada.get("colores", {}).items():
                if nombre in cfg["colores"]:
                    cfg["colores"][nombre].update(rango)
            print(f"[INFO] Configuración cargada de {ruta}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[AVISO] No se pudo leer {ruta}: {e}. Se usan valores por defecto.")
    return cfg


def guardar_config(cfg, ruta=ARCHIVO_CONFIG):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Configuración guardada en {ruta}")


def impar(n, minimo=1):
    """Los kernels de OpenCV deben ser impares (tienen un píxel central)."""
    n = max(minimo, int(n))
    return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# 2. PREPROCESAMIENTO Y MÁSCARAS DE COLOR
# ---------------------------------------------------------------------------
def preprocesar(frame, cfg):
    """Redimensiona, suaviza y convierte a HSV."""
    alto, ancho = frame.shape[:2]
    escala = cfg["ancho_proceso"] / float(ancho)
    if abs(escala - 1.0) > 0.01:
        frame = cv2.resize(frame, (cfg["ancho_proceso"], int(alto * escala)),
                           interpolation=cv2.INTER_AREA)
    k = impar(cfg["blur"])
    suavizada = cv2.GaussianBlur(frame, (k, k), 0)       # reduce ruido del sensor
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)       # BGR -> HSV
    return frame, hsv


def mascara_color(hsv, rango):
    """Crea la máscara binaria (0/255) de un color a partir de su rango HSV.

    cv2.inRange pone en 255 los píxeles cuyos tres canales están dentro del
    rango y en 0 los demás. Si h_min > h_max (caso del rojo) se combinan dos
    rangos con un OR: [h_min, 179] y [0, h_max].
    """
    s0, s1, v0, v1 = rango["s_min"], rango["s_max"], rango["v_min"], rango["v_max"]
    h0, h1 = rango["h_min"], rango["h_max"]
    if h0 <= h1:
        return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    m1 = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
    m2 = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
    return cv2.bitwise_or(m1, m2)


def limpiar_mascara(mascara, cfg):
    """Operaciones morfológicas para eliminar ruido.

    - APERTURA (erosión seguida de dilatación): borra puntos blancos aislados
      y separa regiones unidas por hilos delgados, sin encoger las figuras.
    - CIERRE (dilatación seguida de erosión): rellena huecos negros pequeños
      dentro de la figura (reflejos, brillos, texturas).
    """
    k = impar(cfg["kernel"], 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=int(cfg["iter_apertura"]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=int(cfg["iter_cierre"]))
    return m


# ---------------------------------------------------------------------------
# 3. VALIDACIÓN Y CLASIFICACIÓN DE CONTORNOS
# ---------------------------------------------------------------------------
def toca_borde(contorno, forma_img, margen):
    """True si la figura está cortada por el borde de la imagen (forma incompleta)."""
    x, y, w, h = cv2.boundingRect(contorno)
    alto, ancho = forma_img[:2]
    return x <= margen or y <= margen or x + w >= ancho - margen or y + h >= alto - margen


def pureza_color(contorno, mascara_original):
    """Fracción de píxeles dentro del contorno que realmente son del color.

    Se usa la máscara ANTES de la morfología: si el cierre "inventó" mucha
    área para unir manchas sueltas, la pureza baja y la figura se descarta.
    """
    relleno = np.zeros(mascara_original.shape, np.uint8)
    cv2.drawContours(relleno, [contorno], -1, 255, thickness=cv2.FILLED)
    total = cv2.countNonZero(relleno)
    if total == 0:
        return 0.0
    coinciden = cv2.countNonZero(cv2.bitwise_and(relleno, mascara_original))
    return coinciden / float(total)


def clasificar_forma(contorno, area, cfg):
    """Decide la forma geométrica. Devuelve (nombre, confianza) o (None, motivo)."""
    perimetro = cv2.arcLength(contorno, True)
    if perimetro == 0:
        return None, "perimetro 0"

    # Circularidad: 4·π·Área / Perímetro². Vale 1 en un círculo perfecto,
    # ≈0.785 en un cuadrado y ≈0.60 en un triángulo equilátero.
    circularidad = 4.0 * math.pi * area / (perimetro ** 2)

    # Aproximación poligonal (algoritmo de Douglas-Peucker): reduce el contorno
    # a pocos vértices. epsilon = distancia máxima permitida entre el contorno
    # original y el polígono simplificado.
    aprox = cv2.approxPolyDP(contorno, cfg["epsilon"] * perimetro, True)
    vertices = len(aprox)

    # --- TRIÁNGULO -------------------------------------------------------
    if vertices == 3:
        area_tri, _ = cv2.minEnclosingTriangle(contorno)
        ajuste = area / area_tri if area_tri > 0 else 0
        if ajuste >= cfg["ajuste_min"]:
            return "triangulo", ajuste
        return None, f"triangulo con ajuste bajo ({ajuste:.2f})"

    # --- CUADRILÁTEROS ----------------------------------------------------
    if vertices == 4 and cv2.isContourConvex(aprox):
        (cx, cy), (w, h), angulo = cv2.minAreaRect(contorno)   # rectángulo girado
        if w == 0 or h == 0:
            return None, "rectangulo degenerado"
        ajuste = area / (w * h)                 # qué tanto llena su rectángulo
        if ajuste < cfg["ajuste_min"] + 0.05:    # exigencia un poco mayor
            return None, f"cuadrilatero irregular ({ajuste:.2f})"
        relacion = min(w, h) / max(w, h)          # 1.0 = lados iguales
        if abs(1.0 - relacion) <= cfg["tolerancia_cuadrado"]:
            return "cuadrado", ajuste
        return "rectangulo", ajuste

    # --- CÍRCULO ----------------------------------------------------------
    if vertices >= 6 and circularidad >= cfg["circularidad_min"]:
        (_, _), radio = cv2.minEnclosingCircle(contorno)
        ajuste = area / (math.pi * radio * radio) if radio > 0 else 0
        if ajuste >= cfg["ajuste_min"]:
            return "circulo", ajuste
        return None, f"circulo con ajuste bajo ({ajuste:.2f})"

    # Cualquier otro caso (5 vértices, óvalos, manchas) es DUDOSO -> se ignora
    return None, f"forma dudosa ({vertices} vertices, circ {circularidad:.2f})"


def detectar_en_color(nombre, hsv, forma_img, cfg):
    """Pipeline completo para un color. Devuelve (detecciones, mascara_limpia)."""
    mascara_cruda = mascara_color(hsv, cfg["colores"][nombre])
    mascara = limpiar_mascara(mascara_cruda, cfg)

    # RETR_EXTERNAL: sólo contornos exteriores (ignora huecos internos).
    # CHAIN_APPROX_SIMPLE: guarda sólo los puntos esquina de tramos rectos.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detecciones = []
    for c in contornos:
        area = cv2.contourArea(c)

        # Filtro 1: área mínima y máxima
        if area < cfg["area_min"] or area > cfg["area_max"]:
            continue
        # Filtro 2: figura cortada por el borde de la imagen
        if toca_borde(c, forma_img, cfg["margen_borde"]):
            continue
        # Filtro 3: solidez = área / área de la envolvente convexa.
        # Las figuras geométricas básicas son convexas -> solidez ≈ 1.
        # Manchas irregulares, sombras o dos objetos pegados -> solidez baja.
        envolvente = cv2.convexHull(c)
        area_env = cv2.contourArea(envolvente)
        solidez = area / area_env if area_env > 0 else 0
        if solidez < cfg["solidez_min"]:
            continue
        # Filtro 4: pureza de color dentro del contorno
        if pureza_color(c, mascara_cruda) < cfg["pureza_min"]:
            continue
        # Filtro 5: clasificación geométrica (descarta formas dudosas)
        forma, confianza = clasificar_forma(c, area, cfg)
        if forma is None:
            continue

        M = cv2.moments(c)                    # momentos -> centroide
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        detecciones.append({
            "color": nombre, "forma": forma, "contorno": c,
            "centro": (cx, cy), "area": area, "solidez": solidez,
            "confianza": confianza,
        })
    return detecciones, mascara


def detectar_figuras(frame, cfg):
    """Procesa un cuadro completo. Devuelve (frame_redimensionado, detecciones, mascaras)."""
    frame, hsv = preprocesar(frame, cfg)
    todas, mascaras = [], {}
    for nombre in NOMBRES_COLOR:
        dets, mascaras[nombre] = detectar_en_color(nombre, hsv, frame.shape, cfg)
        todas.extend(dets)
    return frame, todas, mascaras


# ---------------------------------------------------------------------------
# 4. ESTABILIDAD TEMPORAL (SEGUIMIENTO SENCILLO)
# ---------------------------------------------------------------------------
class Seguidor:
    """Asocia detecciones entre cuadros consecutivos por color y cercanía.

    Una figura sólo se considera CONFIRMADA cuando se ha visto en
    'cuadros_confirmar' cuadros seguidos. La forma mostrada es la más votada
    en los últimos cuadros, lo que evita que "parpadee" entre cuadrado y
    rectángulo, por ejemplo.
    """

    def __init__(self):
        self.pistas = []

    def actualizar(self, detecciones, cfg):
        for p in self.pistas:
            p["vista"] = False

        for d in detecciones:
            radio = max(30.0, 0.6 * math.sqrt(d["area"]))  # distancia de asociación
            mejor, mejor_dist = None, radio
            for p in self.pistas:
                if p["color"] != d["color"] or p["vista"]:
                    continue
                dist = math.dist(p["centro"], d["centro"])
                if dist < mejor_dist:
                    mejor, mejor_dist = p, dist
            if mejor is None:
                mejor = {"color": d["color"], "aciertos": 0, "fallos": 0,
                         "formas": deque(maxlen=10)}
                self.pistas.append(mejor)
            mejor.update(centro=d["centro"], contorno=d["contorno"], area=d["area"],
                         vista=True, fallos=0)
            mejor["aciertos"] += 1
            mejor["formas"].append(d["forma"])

        for p in self.pistas:
            if not p["vista"]:
                p["fallos"] += 1
                p["aciertos"] = max(0, p["aciertos"] - 1)
        self.pistas = [p for p in self.pistas if p["fallos"] <= cfg["cuadros_olvidar"]]

    def confirmadas(self, cfg):
        res = []
        for p in self.pistas:
            if p["aciertos"] >= cfg["cuadros_confirmar"] and p["fallos"] <= 1:
                forma, votos = Counter(p["formas"]).most_common(1)[0]
                if votos >= len(p["formas"]) * 0.6:      # mayoría clara
                    res.append({**p, "forma": forma})
        return res


# ---------------------------------------------------------------------------
# 5. VISUALIZACIÓN
# ---------------------------------------------------------------------------
NOMBRE_BONITO = {"triangulo": "Triangulo", "cuadrado": "Cuadrado",
                 "rectangulo": "Rectangulo", "circulo": "Circulo"}


def dibujar(frame, figuras):
    salida = frame.copy()
    for f in figuras:
        col = COLOR_DIBUJO[f["color"]]
        cv2.drawContours(salida, [f["contorno"]], -1, col, 3)
        cx, cy = f["centro"]
        cv2.circle(salida, (cx, cy), 4, (255, 255, 255), -1)
        texto = f"{NOMBRE_BONITO[f['forma']]} {f['color']}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x, y = cx - tw // 2, cy - 12
        cv2.rectangle(salida, (x - 4, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(salida, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    cv2.putText(salida, f"Figuras confirmadas: {len(figuras)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return salida


def mosaico_mascaras(mascaras, ancho=320):
    """Une las tres máscaras en una sola imagen para mostrarlas."""
    paneles = []
    for nombre in NOMBRES_COLOR:
        m = mascaras[nombre]
        alto = int(m.shape[0] * ancho / m.shape[1])
        p = cv2.cvtColor(cv2.resize(m, (ancho, alto)), cv2.COLOR_GRAY2BGR)
        cv2.putText(p, nombre, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLOR_DIBUJO[nombre], 2, cv2.LINE_AA)
        paneles.append(p)
    return np.hstack(paneles)


# ---------------------------------------------------------------------------
# 6. PANEL DE CONFIGURACIÓN CON BARRAS DESLIZANTES (TRACKBARS)
# ---------------------------------------------------------------------------
VENTANA_AJUSTES = "Ajustes"
# (nombre de la barra, clave en cfg, valor máximo, factor de escala)
BARRAS_GENERALES = [
    ("Area min",        "area_min",          20000, 1),
    ("Area max /100",   "area_max",          3000,  100),
    ("Solidez %",       "solidez_min",       100,   0.01),
    ("Pureza %",        "pureza_min",        100,   0.01),
    ("Kernel",          "kernel",            21,    1),
    ("Apertura it",     "iter_apertura",     6,     1),
    ("Cierre it",       "iter_cierre",       6,     1),
    ("Blur",            "blur",              21,    1),
    ("Epsilon x1000",   "epsilon",           100,   0.001),
    ("Circular %",      "circularidad_min",  100,   0.01),
    ("Ajuste %",        "ajuste_min",        100,   0.01),
    ("Cuadros conf",    "cuadros_confirmar", 20,    1),
]
BARRAS_HSV = [("H min", "h_min", 179), ("H max", "h_max", 179),
              ("S min", "s_min", 255), ("S max", "s_max", 255),
              ("V min", "v_min", 255), ("V max", "v_max", 255)]


def _nada(_):
    pass


class PanelAjustes:
    def __init__(self, cfg):
        self.color_actual = 0
        cv2.namedWindow(VENTANA_AJUSTES, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(VENTANA_AJUSTES, 420, 720)
        cv2.createTrackbar("Color 0R 1V 2A", VENTANA_AJUSTES, 0, 2, _nada)
        for etiqueta, _, maximo in BARRAS_HSV:
            cv2.createTrackbar(etiqueta, VENTANA_AJUSTES, 0, maximo, _nada)
        for etiqueta, _, maximo, _ in BARRAS_GENERALES:
            cv2.createTrackbar(etiqueta, VENTANA_AJUSTES, 0, maximo, _nada)
        self.escribir(cfg)

    def escribir(self, cfg):
        """Copia los valores de cfg a las barras."""
        cv2.setTrackbarPos("Color 0R 1V 2A", VENTANA_AJUSTES, self.color_actual)
        self._escribir_hsv(cfg)
        for etiqueta, clave, _, escala in BARRAS_GENERALES:
            cv2.setTrackbarPos(etiqueta, VENTANA_AJUSTES, int(round(cfg[clave] / escala)))

    def _escribir_hsv(self, cfg):
        rango = cfg["colores"][NOMBRES_COLOR[self.color_actual]]
        for etiqueta, clave, _ in BARRAS_HSV:
            cv2.setTrackbarPos(etiqueta, VENTANA_AJUSTES, rango[clave])

    def cambiar_color(self, cfg, indice=None):
        self.color_actual = (self.color_actual + 1) % 3 if indice is None else indice
        cv2.setTrackbarPos("Color 0R 1V 2A", VENTANA_AJUSTES, self.color_actual)
        self._escribir_hsv(cfg)
        print(f"[INFO] Editando rangos HSV de: {NOMBRES_COLOR[self.color_actual]}")

    def leer(self, cfg):
        """Lee las barras y actualiza cfg en vivo."""
        sel = cv2.getTrackbarPos("Color 0R 1V 2A", VENTANA_AJUSTES)
        if sel != self.color_actual:          # el usuario movió el selector
            self.cambiar_color(cfg, sel)
            return
        rango = cfg["colores"][NOMBRES_COLOR[self.color_actual]]
        for etiqueta, clave, _ in BARRAS_HSV:
            rango[clave] = cv2.getTrackbarPos(etiqueta, VENTANA_AJUSTES)
        for etiqueta, clave, _, escala in BARRAS_GENERALES:
            valor = cv2.getTrackbarPos(etiqueta, VENTANA_AJUSTES) * escala
            cfg[clave] = int(valor) if escala >= 1 else round(valor, 3)
        cfg["kernel"] = impar(cfg["kernel"], 3)
        cfg["blur"] = impar(cfg["blur"])
        cfg["cuadros_confirmar"] = max(1, cfg["cuadros_confirmar"])
        cfg["area_max"] = max(cfg["area_max"], cfg["area_min"] + 1)


# ---------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------
def abrir_camara(indice):
    cam = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
    if not cam.isOpened():
        cam = cv2.VideoCapture(indice)          # reintento con backend por defecto
    if not cam.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir la cámara {indice}.")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def main():
    parser = argparse.ArgumentParser(description="Detector de figuras rojas, verdes y azules")
    parser.add_argument("--camara", type=int, default=0, help="índice de la cámara (0, 1, ...)")
    parser.add_argument("--imagen", type=str, help="procesar una imagen fija en lugar de la cámara")
    args = parser.parse_args()

    cfg = cargar_config()
    panel = PanelAjustes(cfg)
    seguidor = Seguidor()
    ver_mascaras, pausa = True, False

    imagen_fija = None
    cam = None
    if args.imagen:
        imagen_fija = cv2.imread(args.imagen)
        if imagen_fija is None:
            raise SystemExit(f"[ERROR] No se pudo leer la imagen {args.imagen}")
    else:
        cam = abrir_camara(args.camara)

    print(__doc__.split("Teclas:")[1].split("Uso:")[0])
    frame = None
    while True:
        if not pausa or frame is None:
            if imagen_fija is not None:
                frame = imagen_fija.copy()
            else:
                ok, frame = cam.read()
                if not ok:
                    print("[ERROR] No se recibió imagen de la cámara.")
                    break

        panel.leer(cfg)
        reducido, detecciones, mascaras = detectar_figuras(frame, cfg)
        seguidor.actualizar(detecciones, cfg)
        figuras = seguidor.confirmadas(cfg)

        cv2.imshow("Deteccion de figuras", dibujar(reducido, figuras))
        if ver_mascaras:
            cv2.imshow("Mascaras (rojo | verde | azul)", mosaico_mascaras(mascaras))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla == ord("m"):
            ver_mascaras = not ver_mascaras
            if not ver_mascaras:
                cv2.destroyWindow("Mascaras (rojo | verde | azul)")
        elif tecla == ord("c"):
            panel.cambiar_color(cfg)
        elif tecla == ord("s"):
            guardar_config(cfg)
        elif tecla == ord("r"):
            cfg = copy.deepcopy(CONFIG_POR_DEFECTO)
            panel.escribir(cfg)
            print("[INFO] Valores por defecto restaurados.")
        elif tecla == ord("p"):
            pausa = not pausa

    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
