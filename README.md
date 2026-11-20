# Detección de figuras geométricas por color con OpenCV

Proyecto didáctico del curso de **Programación en Python** (unidad de visión artificial). Detecta en tiempo real, con una cámara web, figuras geométricas de colores (triángulos, cuadrados, rectángulos y círculos) usando segmentación en el espacio de color HSV y análisis de contornos con OpenCV.

Incluye dos programas:

| Archivo | Para qué sirve |
|---|---|
| `detector_figuras.py` | Detector de figuras **rojas, verdes y azules** en tiempo real. |
| `calibrador_colores.py` | Herramienta para ver cómo la cámara "ve" los colores (**rojo, azul y amarillo**) y ajustar los rangos HSV antes de usarlos en un detector. |
| `Detección de figuras por color con OpenCV.pdf` | Documento de apoyo con la explicación teórica y práctica. |

## Requisitos

- Python 3.8 o superior
- Una cámara web (o una imagen de prueba)
- Bibliotecas: `opencv-python` y `numpy`

```bash
pip install -r requirements.txt
```

## Detector de figuras (`detector_figuras.py`)

### Flujo de procesamiento (por cada cuadro de la cámara)

1. **Captura**: se lee un cuadro (frame) de la webcam.
2. **Preprocesamiento**: suavizado con filtro Gaussiano para eliminar ruido.
3. **Conversión HSV**: de BGR a HSV (tono, saturación, brillo).
4. **Umbralización**: una máscara binaria por color (rojo, verde, azul).
5. **Morfología**: apertura (quita puntos) y cierre (rellena huecos).
6. **Contornos**: se buscan las siluetas blancas de cada máscara.
7. **Validación**: área, solidez, pureza de color, bordes y convexidad.
8. **Clasificación**: triángulo, cuadrado, rectángulo o círculo. Si la forma es dudosa, se descarta.
9. **Estabilidad**: una figura sólo se muestra si aparece en varios cuadros seguidos (evita falsos positivos fugaces).
10. **Visualización**: se dibujan sólo las figuras confirmadas.

### Uso

```bash
python detector_figuras.py                     # cámara 0
python detector_figuras.py --camara 1          # otra cámara
python detector_figuras.py --imagen foto.jpg   # probar con una imagen
```

### Teclas

| Tecla | Acción |
|---|---|
| `q` / `ESC` | Salir |
| `m` | Mostrar / ocultar las máscaras de cada color |
| `c` | Cambiar el color cuyos rangos HSV se editan (también con barra) |
| `s` | Guardar la configuración en `config_figuras.json` |
| `r` | Recargar la configuración por defecto |
| `p` | Pausar / reanudar |

## Calibrador de colores (`calibrador_colores.py`)

Muestra qué píxeles pasan el filtro de cada color y pinta de negro todo lo demás. Permite ajustar los rangos HSV con barras, auto-calibrar seleccionando un objeto con el mouse y medir los valores HSV/RGB exactos de una zona.

### Ventanas

- **Camara**: imagen original con las regiones detectadas, su color y área, y una mira central que indica el HSV de ese punto.
- **Detectado**: sólo los píxeles que pasan el filtro.
- **Mascaras**: máscara blanco/negro de cada color.
- **Ajustes**: barras para mover los rangos HSV del color seleccionado.

En consola informa, una vez por segundo, qué colores detecta, cuántas regiones, el área mayor y el porcentaje de la imagen.

### Uso

```bash
python calibrador_colores.py              # cámara 0
python calibrador_colores.py --camara 1
```

### Teclas

| Tecla | Acción |
|---|---|
| `1` / `2` / `3` | Seleccionar ROJO / AZUL / AMARILLO para editar |
| `a` | Auto-calibrar: dibuja un rectángulo sobre un objeto del color y pulsa ENTER |
| `m` | Medir una zona: imprime H, S, V (mín, máx, promedio) y RGB |
| clic izquierdo | Imprime el HSV y RGB del punto y a qué color corresponde |
| `s` | Guardar rangos en `calibracion_colores.json` |
| `r` | Restaurar rangos por defecto |
| `p` | Pausar / reanudar |
| `q` / `ESC` | Salir |

## Notas sobre HSV en OpenCV

- `H` (tono) va de 0 a 179; `S` y `V` de 0 a 255.
- El rojo está en ambos extremos del círculo de tono, por eso su rango "da la vuelta" por 0 (por ejemplo, `h_min = 170` y `h_max = 8`).
- La iluminación cambia mucho los valores de `S` y `V`: conviene calibrar en el mismo lugar donde se usará el detector.

## Autor

**Arturo Elfego Meneses Flores**
Universidad Tecnológica de la Huasteca Hidalguense (UTHH)

## 13. Licencia

[MIT](LICENSE) © 2026 Arturo Elfego Meneses Flores — DUAL ARD / UTHH.
