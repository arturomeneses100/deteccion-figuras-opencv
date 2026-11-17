#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 CALIBRADOR DE COLOR (ROJO, AZUL Y AMARILLO) PARA CÁMARA WEB
==============================================================================
 Sirve para ver CÓMO tu cámara "ve" los colores y ajustar los rangos HSV
 antes de usarlos en un detector. Todo lo que no sea rojo, azul o amarillo
 se pinta de NEGRO (se ignora).

 Ventanas:
   - "Camara"   : imagen original + regiones detectadas con su color y área,
                  y una mira al centro que indica el HSV de ese punto.
   - "Detectado": sólo los píxeles que pasan el filtro (lo demás en negro).
   - "Mascaras" : máscara blanco/negro de cada color (rojo | azul | amarillo).
   - "Ajustes"  : barras para mover los rangos HSV del color seleccionado.

 Salida en consola: una vez por segundo dice qué colores está detectando,
 cuántas regiones, el área mayor y el porcentaje de la imagen.

 Teclas:
   1 / 2 / 3 : seleccionar ROJO / AZUL / AMARILLO para editar
   a         : AUTO-CALIBRAR el color seleccionado: dibuja con el mouse un
               rectángulo sobre un objeto de ese color y pulsa ENTER
   m         : MEDIR una zona: selecciona un rectángulo y ENTER; imprime los
               valores exactos H, S, V (mín, máx, promedio) y RGB, sin filtrar
   clic izq. : imprime el HSV y RGB exactos del punto y qué color le corresponde
   s         : guardar rangos en calibracion_colores.json
   r         : restaurar rangos por defecto
   p         : pausar / reanudar
   q / ESC   : salir

 Uso:
   python calibrador_colores.py            # cámara 0
   python calibrador_colores.py --camara 1
==============================================================================
"""

import argparse
import copy
import json
import os
import time

import cv2
import numpy as np

ARCHIVO = "calibracion_colores.json"

# H: 0-179, S y V: 0-255.  Si h_min > h_max el rango da la vuelta por 0 (rojo).
RANGOS_POR_DEFECTO = {
    "rojo":     {"h_min": 170, "h_max": 8,   "s_min": 120, "s_max": 255, "v_min": 70, "v_max": 255},
    "azul":     {"h_min": 95,  "h_max": 128, "s_min": 120, "s_max": 255, "v_min": 50, "v_max": 255},
    "amarillo": {"h_min": 20,  "h_max": 34,  "s_min": 100, "s_max": 255, "v_min": 100, "v_max": 255},
}
GENERALES_POR_DEFECTO = {"area_min": 800, "kernel": 5, "blur": 5}

COLORES = ["rojo", "azul", "amarillo"]
DIBUJO = {"rojo": (0, 0, 255), "azul": (255, 80, 0), "amarillo": (0, 220, 255)}


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def cargar():
    rangos = copy.deepcopy(RANGOS_POR_DEFECTO)
    generales = dict(GENERALES_POR_DEFECTO)
    if os.path.exists(ARCHIVO):
        try:
            with open(ARCHIVO, encoding="utf-8") as f:
                datos = json.load(f)
            for c in COLORES:
                rangos[c].update(datos.get("colores", {}).get(c, {}))
            generales.update({k: datos[k] for k in generales if k in datos})
            print(f"[INFO] Calibración cargada de {ARCHIVO}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[AVISO] No se pudo leer {ARCHIVO}: {e}")
    return rangos, generales


def guardar(rangos, generales):
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump({"colores": rangos, **generales}, f, indent=2, ensure_ascii=False)
    print(f"\n[GUARDADO] {ARCHIVO}")
    for c in COLORES:
        print(f"   {c:9s}: {rangos[c]}")


def impar(n, minimo=1):
    n = max(minimo, int(n))
    return n if n % 2 else n + 1


# ---------------------------------------------------------------------------
# Procesamiento
# ---------------------------------------------------------------------------
def mascara(hsv, r):
    """Máscara binaria de un color. Soporta el rango 'que da la vuelta' del rojo."""
    lo_s, hi_s, lo_v, hi_v = r["s_min"], r["s_max"], r["v_min"], r["v_max"]
    if r["h_min"] <= r["h_max"]:
        return cv2.inRange(hsv, (r["h_min"], lo_s, lo_v), (r["h_max"], hi_s, hi_v))
    return cv2.bitwise_or(cv2.inRange(hsv, (r["h_min"], lo_s, lo_v), (179, hi_s, hi_v)),
                          cv2.inRange(hsv, (0, lo_s, lo_v), (r["h_max"], hi_s, hi_v)))


def color_de_pixel(h, s, v, rangos):
    """Devuelve qué color(es) corresponden a un píxel HSV, o 'ninguno'."""
    px = np.uint8([[[h, s, v]]])
    nombres = [c for c in COLORES if mascara(px, rangos[c])[0, 0] > 0]
    return ", ".join(nombres) if nombres else "ninguno (ignorado)"


def procesar(frame, rangos, generales):
    """Devuelve hsv, máscaras limpias y lista de regiones por color."""
    k = impar(generales["blur"])
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (k, k), 0), cv2.COLOR_BGR2HSV)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (impar(generales["kernel"], 3),) * 2)
    mascaras, regiones = {}, {}
    for c in COLORES:
        m = mascara(hsv, rangos[c])
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)     # quita puntos sueltos
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)    # rellena huecos
        mascaras[c] = m
        conts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regiones[c] = sorted([x for x in conts if cv2.contourArea(x) >= generales["area_min"]],
                             key=cv2.contourArea, reverse=True)
    return hsv, mascaras, regiones


def estadisticas(hsv, frame, mascara_zona=None, envuelve=False):
    """Valores EXACTOS de los píxeles de una zona: H, S, V (mín, máx, promedio)
    y color promedio en RGB. Si mascara_zona es None se usan todos los píxeles.

    Para el rojo (envuelve=True) el tono se desplaza +90 para calcular, porque
    sus valores están en ambos extremos (ej. 175 y 3) y su promedio simple
    daría 89, que es VERDE. El rango se reporta como 'h_min->h_max'.
    """
    sel = (slice(None),) if mascara_zona is None else (mascara_zona > 0,)
    H = hsv[..., 0][sel].astype(int).ravel()
    S = hsv[..., 1][sel].astype(int).ravel()
    V = hsv[..., 2][sel].astype(int).ravel()
    B, G, R = (frame[..., i][sel].astype(float).ravel() for i in range(3))
    if H.size == 0:
        return None
    if envuelve:
        hd = (H + 90) % 180
        h_min, h_max = (int(hd.min()) - 90) % 180, (int(hd.max()) - 90) % 180
        h_prom = (int(round(hd.mean())) - 90) % 180
    else:
        h_min, h_max, h_prom = int(H.min()), int(H.max()), int(round(H.mean()))
    return {
        "pixeles": int(H.size),
        "H": (h_min, h_max, h_prom), "S": (int(S.min()), int(S.max()), int(round(S.mean()))),
        "V": (int(V.min()), int(V.max()), int(round(V.mean()))),
        "RGB": (int(R.mean()), int(G.mean()), int(B.mean())),
    }


def stats_region(hsv, frame, cont, color):
    """Estadísticas exactas sólo de los píxeles dentro de un contorno."""
    x, y, w, h = cv2.boundingRect(cont)
    zona = np.zeros((h, w), np.uint8)
    cv2.drawContours(zona, [cont - (x, y)], -1, 255, cv2.FILLED)
    return estadisticas(hsv[y:y + h, x:x + w], frame[y:y + h, x:x + w], zona, color == "rojo")


def formato_stats(st):
    h, s, v = st["H"], st["S"], st["V"]
    return (f"H {h[0]}->{h[1]} (prom {h[2]})  S {s[0]}-{s[1]} (prom {s[2]})  "
            f"V {v[0]}-{v[1]} (prom {v[2]})  RGB prom {st['RGB']}  [{st['pixeles']} px]")


def auto_rango(hsv_roi, color):
    """Calcula un rango HSV a partir de una zona seleccionada (percentiles 5-95).

    Para el rojo, los tonos están cerca de 0 y de 179 a la vez; se desplaza el
    tono +90 (módulo 180) para que queden juntos, se calcula y se regresa.
    """
    h = hsv_roi[..., 0].astype(int).ravel()
    s = hsv_roi[..., 1].ravel()
    v = hsv_roi[..., 2].ravel()
    # descartar píxeles casi grises u oscuros de la selección
    validos = (s > 40) & (v > 40)
    if validos.sum() < 50:
        print("[AVISO] La zona seleccionada casi no tiene color (gris/oscuro). No se cambió nada.")
        return None
    h, s, v = h[validos], s[validos], v[validos]

    envuelve = color == "rojo" or (np.percentile(h, 95) - np.percentile(h, 5) > 90)
    if envuelve:
        hd = (h + 90) % 180
        h_min, h_max = [(int(np.percentile(hd, p)) - 90) % 180 for p in (5, 95)]
    else:
        h_min, h_max = int(np.percentile(h, 5)), int(np.percentile(h, 95))
    margen = 4
    if envuelve:      # margen circular: 0-4 pasa a 176->8 (sí cruza por 0)
        h_min, h_max = (h_min - margen) % 180, (h_max + margen) % 180
    else:
        h_min, h_max = max(0, h_min - margen), min(179, h_max + margen)
    return {
        "h_min": h_min, "h_max": h_max,
        "s_min": max(0, int(np.percentile(s, 5)) - 25), "s_max": 255,
        "v_min": max(0, int(np.percentile(v, 5)) - 30), "v_max": 255,
    }


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def texto(img, t, org, color=(255, 255, 255), esc=0.55):
    (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, esc, 2)
    x, y = org
    cv2.rectangle(img, (x - 3, y - h - 5), (x + w + 3, y + 5), (0, 0, 0), -1)
    cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, color, 2, cv2.LINE_AA)


def dibujar(frame, hsv, regiones, rangos, editando):
    out = frame.copy()
    total = frame.shape[0] * frame.shape[1]
    for c in COLORES:
        for cont in regiones[c]:
            area = cv2.contourArea(cont)
            x, y, w, h = cv2.boundingRect(cont)
            st = stats_region(hsv, frame, cont, c)
            cv2.drawContours(out, [cont], -1, DIBUJO[c], 2)
            # etiquetas debajo de la región (o arriba si no caben)
            y1 = y + h + 18 if y + h + 50 < frame.shape[0] - 30 else max(80, y - 26)
            texto(out, f"{c.upper()} {int(area)}px ({100 * area / total:.1f}%)", (x, y1), DIBUJO[c], 0.5)
            texto(out, f"H{st['H'][0]}->{st['H'][1]} S{st['S'][0]}-{st['S'][1]} V{st['V'][0]}-{st['V'][1]}",
                  (x, y1 + 20), (255, 255, 255), 0.45)

    # Mira central con lectura HSV
    alto, ancho = frame.shape[:2]
    cx, cy = ancho // 2, alto // 2
    cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
    h, s, v = (int(x) for x in hsv[cy, cx])
    b, g, rr = (int(x) for x in frame[cy, cx])
    texto(out, f"Centro HSV=({h},{s},{v}) RGB=({rr},{g},{b}) -> {color_de_pixel(h, s, v, rangos)}",
          (10, alto - 12), esc=0.5)

    resumen = "  ".join(f"{c.upper()}:{len(regiones[c])}" for c in COLORES)
    texto(out, resumen, (10, 22))
    r = rangos[editando]
    texto(out, f"Editando [{editando.upper()}] H {r['h_min']}-{r['h_max']}  "
               f"S {r['s_min']}-{r['s_max']}  V {r['v_min']}-{r['v_max']}", (10, 48), DIBUJO[editando], 0.5)
    return out


def solo_detectado(frame, mascaras):
    """Imagen donde todo lo que no es rojo/azul/amarillo queda negro."""
    total = np.zeros(frame.shape[:2], np.uint8)
    for m in mascaras.values():
        total = cv2.bitwise_or(total, m)
    return cv2.bitwise_and(frame, frame, mask=total)


def mosaico(mascaras, ancho=300):
    paneles = []
    for c in COLORES:
        m = mascaras[c]
        p = cv2.cvtColor(cv2.resize(m, (ancho, int(m.shape[0] * ancho / m.shape[1]))), cv2.COLOR_GRAY2BGR)
        texto(p, c, (8, 22), DIBUJO[c], 0.6)
        paneles.append(p)
    return np.hstack(paneles)


def reporte_consola(hsv, frame, regiones, max_por_color=3):
    """Imprime los valores exactos de cada región detectada (hasta 3 por color)."""
    total = frame.shape[0] * frame.shape[1]
    hora = time.strftime("%H:%M:%S")
    lineas = []
    for c in COLORES:
        for i, cont in enumerate(regiones[c][:max_por_color], 1):
            area = cv2.contourArea(cont)
            x, y, w, h = cv2.boundingRect(cont)
            st = stats_region(hsv, frame, cont, c)
            lineas.append(f"  {c.upper():8s} #{i} area {int(area)} px ({100 * area / total:.1f}%) "
                          f"centro ({x + w // 2},{y + h // 2})\n             {formato_stats(st)}")
        if len(regiones[c]) > max_por_color:
            lineas.append(f"  {c.upper():8s} ... y {len(regiones[c]) - max_por_color} region(es) mas")
    print(f"\n[{hora}] " + ("Detectando:" if lineas else "Nada detectado (todo ignorado)"))
    for l in lineas:
        print(l)


# ---------------------------------------------------------------------------
# Panel de ajustes
# ---------------------------------------------------------------------------
VENT = "Ajustes"
BARRAS_HSV = [("H min", "h_min", 179), ("H max", "h_max", 179), ("S min", "s_min", 255),
              ("S max", "s_max", 255), ("V min", "v_min", 255), ("V max", "v_max", 255)]
BARRAS_GEN = [("Area min", "area_min", 20000), ("Kernel", "kernel", 21), ("Blur", "blur", 21)]


def crear_panel(rangos, generales, editando):
    cv2.namedWindow(VENT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENT, 420, 380)
    for et, _, mx in BARRAS_HSV + BARRAS_GEN:
        cv2.createTrackbar(et, VENT, 0, mx, lambda _: None)
    escribir_panel(rangos, generales, editando)


def escribir_panel(rangos, generales, editando):
    for et, k, _ in BARRAS_HSV:
        cv2.setTrackbarPos(et, VENT, rangos[editando][k])
    for et, k, _ in BARRAS_GEN:
        cv2.setTrackbarPos(et, VENT, int(generales[k]))
    cv2.setWindowTitle(VENT, f"Ajustes - editando {editando.upper()}")


def leer_panel(rangos, generales, editando):
    for et, k, _ in BARRAS_HSV:
        rangos[editando][k] = cv2.getTrackbarPos(et, VENT)
    for et, k, _ in BARRAS_GEN:
        generales[k] = cv2.getTrackbarPos(et, VENT)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibrador de color rojo/azul/amarillo")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--intervalo", type=float, default=1.0,
                    help="segundos entre reportes en consola (default 1)")
    args = ap.parse_args()
    generales_intervalo = args.intervalo

    cam = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit(f"[ERROR] No se pudo abrir la cámara {args.camara}")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    rangos, generales = cargar()
    editando = "rojo"
    crear_panel(rangos, generales, editando)
    estado = {"hsv": None}

    def clic(evento, x, y, flags, _):
        if evento == cv2.EVENT_LBUTTONDOWN and estado["hsv"] is not None:
            h, s, v = (int(n) for n in estado["hsv"][y, x])
            b, g, r = (int(n) for n in estado["frame"][y, x])
            print(f"[CLIC] ({x},{y}) HSV=({h},{s},{v})  RGB=({r},{g},{b})  -> {color_de_pixel(h, s, v, rangos)}")

    cv2.namedWindow("Camara")
    cv2.setMouseCallback("Camara", clic)
    print(__doc__.split("Teclas:")[1].split("Uso:")[0])

    pausa, frame, ultimo_reporte = False, None, 0.0
    while True:
        if not pausa or frame is None:
            ok, frame = cam.read()
            if not ok:
                print("[ERROR] La cámara no entregó imagen.")
                break

        leer_panel(rangos, generales, editando)
        hsv, mascaras, regiones = procesar(frame, rangos, generales)
        estado["hsv"], estado["frame"] = hsv, frame

        cv2.imshow("Camara", dibujar(frame, hsv, regiones, rangos, editando))
        cv2.imshow("Detectado", solo_detectado(frame, mascaras))
        cv2.imshow("Mascaras", mosaico(mascaras))

        if time.time() - ultimo_reporte >= generales_intervalo:
            reporte_consola(hsv, frame, regiones)
            ultimo_reporte = time.time()

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla in (ord("1"), ord("2"), ord("3")):
            editando = COLORES[tecla - ord("1")]
            escribir_panel(rangos, generales, editando)
            print(f"[INFO] Editando {editando.upper()}")
        elif tecla == ord("a"):
            print(f"[AUTO] Selecciona una zona {editando.upper()} y pulsa ENTER (c = cancelar)")
            x, y, w, h = cv2.selectROI("Camara", frame, showCrosshair=True)
            if w > 5 and h > 5:
                nuevo = auto_rango(hsv[y:y + h, x:x + w], editando)
                if nuevo:
                    rangos[editando] = nuevo
                    escribir_panel(rangos, generales, editando)
                    print(f"[AUTO] {editando.upper()} -> {nuevo}")
            cv2.setMouseCallback("Camara", clic)
        elif tecla == ord("m"):
            print("[MEDIR] Selecciona cualquier zona y pulsa ENTER (no cambia los rangos)")
            x, y, w, h = cv2.selectROI("Camara", frame, showCrosshair=True)
            if w > 2 and h > 2:
                zh, zf = hsv[y:y + h, x:x + w], frame[y:y + h, x:x + w]
                normal = estadisticas(zh, zf)
                rojo = estadisticas(zh, zf, envuelve=True)
                print(f"[MEDIR] Zona ({x},{y}) {w}x{h}")
                print(f"   {formato_stats(normal)}")
                print(f"   Si es rojo, tono medido con vuelta por 0: H {rojo['H'][0]}->{rojo['H'][1]} (prom {rojo['H'][2]})")
                sug = auto_rango(zh, editando)
                if sug:
                    print(f"   Rango sugerido para {editando.upper()} (pulsa 'a' para aplicarlo): {sug}")
            cv2.setMouseCallback("Camara", clic)
        elif tecla == ord("s"):
            guardar(rangos, generales)
        elif tecla == ord("r"):
            rangos, generales = copy.deepcopy(RANGOS_POR_DEFECTO), dict(GENERALES_POR_DEFECTO)
            escribir_panel(rangos, generales, editando)
            print("[INFO] Rangos por defecto restaurados")
        elif tecla == ord("p"):
            pausa = not pausa

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
