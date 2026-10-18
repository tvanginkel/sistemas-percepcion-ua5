"""
seguimiento_yolo_final.py - Seguimiento de objetos con YOLO: dirección y velocidad.

A cada objeto se le asigna un ID estable entre fotogramas (tracker ByteTrack)
y con el historial de su caja se miden dos tipos de movimiento:

  1. LATERAL (en el plano de la imagen): flecha hacia donde se mueve, estela
     del recorrido, velocidad (px/s o m/s) y ángulo.
  2. EN PROFUNDIDAD (acercándose o alejándose de la cámara): se mide cuánto
     crece o encoge el objeto por segundo (%/s). Si su tamaño aparente crece
     un 25 %/s, su distancia a la cámara se reduce a ese ritmo. Con eso se
     estima el TIEMPO HASTA EL CONTACTO (TTC = 1 / ritmo de crecimiento), sin
     necesidad de conocer la distancia ni el tamaño real del objeto.
     Se dibujan flechas en las esquinas de la caja: hacia fuera si se acerca,
     hacia dentro si se aleja.

Por defecto SOLO se muestran los objetos en movimiento. Los quietos, los que
tocan el borde de la imagen y los recién aparecidos se ocultan; la tecla 'm'
los muestra u oculta en gris.

El procesamiento se hace siempre a la resolución original del vídeo; el
tamaño de la ventana solo cambia lo que ves en pantalla.

Requisitos:  pip install ultralytics lap

Uso:
    python seguimiento_yolo.py                       -> webcam en vivo (por defecto)
    python seguimiento_yolo.py --fuente video.mp4    -> vídeo
    python seguimiento_yolo.py --vista 1.5           -> ventana al 150 %
    python seguimiento_yolo.py --escala 400          -> velocidad lateral en m/s
    python seguimiento_yolo.py --reduccion 0.5       -> YOLO procesa a la mitad (más rápido)

Opciones:
    --modelo   yolo26n.pt (por defecto), yolo26s.pt...
    --conf     confianza mínima (0-1, por defecto 0.4)
    --vista    tamaño de la ventana respecto al vídeo original (por defecto 1.0)
    --escala   píxeles por metro a la distancia del objeto (opcional). Sin ella
               la velocidad lateral se muestra en píxeles/segundo.
    --tracker  bytetrack.yaml (por defecto) o botsort.yaml
    --reduccion  factor de resolución con el que procesa YOLO (por defecto 1.0 =
               640 px). Con 0.5 procesa a 320 px: bastante más rápido, pero
               detecta peor los objetos pequeños o lejanos. Las cajas siguen en
               coordenadas de la imagen original, así que velocidades y
               umbrales no cambian.

Teclas (con la ventana de vídeo seleccionada):
    m      mostrar / ocultar los objetos quietos
    + / -  agrandar / reducir la ventana
    q      salir
"""

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

VENTANA_S = 0.5        # segundos de historial usados para calcular el movimiento
HISTORIAL_MIN_S = 0.3  # segundos mínimos de historial antes de dar una medida
# Movimiento lateral (px/s). Dos umbrales distintos (histéresis) evitan que un
# objeto lento parpadee entre "quieto" y "en movimiento".
UMBRAL_MOVIL = 40      # px/s: por encima, pasa a "en movimiento"
UMBRAL_QUIETO = 10     # px/s: 25 por debajo, vuelve a "quieto"
# Movimiento en profundidad (ritmo de cambio de tamaño, fracción por segundo)
UMBRAL_PROF_MOVIL = 0.20    # 20 %/s: por encima, se acerca o se aleja
UMBRAL_PROF_QUIETO = 0.12   # 12 %/s: por debajo, deja de considerarse
TTC_ALERTA_S = 2.0     # si el tiempo hasta el contacto baja de esto: alerta roja
OLVIDAR_S = 4.0        # segundos sin ver un ID antes de borrar su historial
LARGO_ESTELA = 30      # puntos de la estela dibujada
MARGEN_BORDE = 4       # px: una caja a menos de esto del borde está "cortada"
VISTA_MIN, VISTA_MAX, VISTA_PASO = 0.25, 3.0, 0.25
IMGSZ_BASE = 640       # tamaño de entrada estándar de YOLO (reducción 1.0)
GRIS = (160, 160, 160)
ROJO = (0, 0, 255)


def parse_args():
    p = argparse.ArgumentParser(description="Seguimiento de objetos con YOLO")
    p.add_argument("--fuente", default="1", help="0 = webcam, o ruta de vídeo")
    p.add_argument("--modelo", default="yolo26n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--vista", type=float, default=1.0,
                   help="Tamaño de la ventana (1.0 = tamaño original)")
    p.add_argument("--escala", type=float, default=None,
                   help="Píxeles por metro (opcional, para mostrar m/s)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--reduccion", type=float, default=1.0,
                   help="Factor de resolución de proceso de YOLO (1.0 = 640 px, 0.5 = 320 px)")
    return p.parse_args()


def color_para_id(track_id):
    """Color fijo y distinto para cada ID."""
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(80, 256, size=3))


def mismo_sentido_min(a, b):
    """
    Si a y b tienen el mismo signo, devuelve el de menor magnitud (con ese
    signo); si no, 0. Es la regla que separa el movimiento real de las
    deformaciones de la caja (oclusiones, piernas que se abren al andar...).
    """
    if a * b <= 0:
        return 0.0
    return math.copysign(min(abs(a), abs(b)), a)


def pendiente(t, valores):
    """Pendiente de la recta de mínimos cuadrados (t ya centrado en 0)."""
    return (t[:, None] * (valores - valores.mean(axis=0))).sum(axis=0) / (t * t).sum()


def estimar_movimiento(historial, t_ahora):
    """
    Devuelve (vx, vy, r, retraso) o None si aún no hay datos suficientes:
      vx, vy  : velocidad lateral en px/s
      r       : ritmo de cambio de tamaño en 1/s (>0 se acerca, <0 se aleja)
      retraso : segundos que la medida va por detrás del fotograma actual

    - Profundidad: se ajusta una recta al logaritmo del ancho y del alto de la
      caja; su pendiente es el ritmo relativo de crecimiento. Solo se acepta si
      ancho y alto cambian en el mismo sentido: si se acerca, crecen los dos;
      si otro objeto lo tapa, solo cambia uno.
    - Lateral: se mide la velocidad de cada borde de la caja, se descuenta la
      parte debida al crecimiento y se exige que los bordes opuestos se muevan
      en el mismo sentido (así una oclusión no parece un desplazamiento).
    """
    muestras = [m for m in historial if t_ahora - m[0] <= VENTANA_S]
    t = np.array([m[0] for m in muestras])
    # Hace falta un mínimo de historial: con muy pocas muestras, el temblor de
    # la caja de un objeto recién aparecido da lecturas falsas.
    if len(muestras) < 3 or np.ptp(t) < HISTORIAL_MIN_S:
        return None
    t = t - t.mean()
    cajas = np.array([m[3:] for m in muestras])           # x1, y1, x2, y2
    ancho = np.maximum(cajas[:, 2] - cajas[:, 0], 1.0)
    alto = np.maximum(cajas[:, 3] - cajas[:, 1], 1.0)

    r_ancho, r_alto = pendiente(t, np.log(np.stack([ancho, alto], axis=1)))
    if r_ancho * r_alto > 0 and max(abs(r_ancho), abs(r_alto)) <= 2 * min(abs(r_ancho), abs(r_alto)):
        r = (r_ancho + r_alto) / 2       # crecen parecido: cambio de escala real
    else:
        r = mismo_sentido_min(r_ancho, r_alto)   # deformación u oclusión: prudente

    v_x1, v_y1, v_x2, v_y2 = pendiente(t, cajas)
    crec_x = r * ancho.mean() / 2      # velocidad de cada borde debida al crecimiento
    crec_y = r * alto.mean() / 2
    vx = mismo_sentido_min(v_x1 + crec_x, v_x2 - crec_x)
    vy = mismo_sentido_min(v_y1 + crec_y, v_y2 - crec_y)
    # La recta describe el movimiento en el centro de la ventana de tiempo, que
    # va 'retraso' segundos por detrás del fotograma actual.
    retraso = t_ahora - np.mean([m[0] for m in muestras])
    return vx, vy, r, retraso


def etiqueta(imagen, texto, posicion, color, escala=0.6):
    """Texto con fondo negro para que se lea sobre cualquier imagen."""
    fuente, grosor = cv2.FONT_HERSHEY_SIMPLEX, 2
    (ancho, alto), base = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = int(posicion[0]), int(posicion[1])
    x = max(5, min(x, imagen.shape[1] - ancho - 5))
    y = max(alto + 6, min(y, imagen.shape[0] - base - 2))
    cv2.rectangle(imagen, (x - 4, y - alto - 5), (x + ancho + 4, y + base + 1),
                  (0, 0, 0), -1)
    cv2.putText(imagen, texto, (x, y), fuente, escala, color, grosor)


def flecha(imagen, inicio, fin, color, grosor=3):
    """Flecha con contorno negro para que destaque sobre cualquier fondo."""
    cv2.arrowedLine(imagen, inicio, fin, (0, 0, 0), grosor + 3, tipLength=0.3)
    cv2.arrowedLine(imagen, inicio, fin, color, grosor, tipLength=0.3)


def flechas_esquinas(imagen, x1, y1, x2, y2, acercandose, color):
    """Flechas diagonales en las esquinas: hacia fuera = se acerca, hacia dentro = se aleja."""
    largo = int(max(15, min(35, 0.15 * min(x2 - x1, y2 - y1))))
    for ex, ey, dx, dy in ((x1, y1, -1, -1), (x2, y1, 1, -1),
                           (x1, y2, -1, 1), (x2, y2, 1, 1)):
        dentro = (ex - dx * largo, ey - dy * largo)      # punto interior
        fuera = (ex + dx * largo // 2, ey + dy * largo // 2)
        if acercandose:
            flecha(imagen, dentro, fuera, color, 2)
        else:
            flecha(imagen, fuera, dentro, color, 2)


def main():
    args = parse_args()
    fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente
    vista = min(max(args.vista, VISTA_MIN), VISTA_MAX)

    # Reloj: en vivo usamos el tiempo real; en un vídeo, el tiempo del propio
    # vídeo (nº de fotograma / fps), para que la velocidad no dependa de lo
    # rápido que procese tu ordenador.
    fps_video = None
    if isinstance(fuente, str):
        cap = cv2.VideoCapture(fuente)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()

    # Resolución de proceso: YOLO redimensiona cada imagen a 'imgsz' antes de
    # analizarla (640 por defecto) y eso es lo que determina su coste; reducir
    # solo la imagen de la cámara no aceleraría nada. Debe ser múltiplo de 32.
    imgsz = max(64, int(round(IMGSZ_BASE * args.reduccion / 32)) * 32)

    print(f"Cargando modelo {args.modelo}...")
    modelo = YOLO(args.modelo)
    print(f"Resolución de proceso: {imgsz} px (reducción {args.reduccion:g})")
    if fuente == 0:
        print("Modo por defecto: seguimiento en vivo con la webcam.")
    unidad = "m/s" if args.escala else "px/s"
    print(f"Velocidad lateral en {unidad}; profundidad en %/s de cambio de tamaño.")
    print("Teclas: 'm' quietos on/off, '+'/'-' tamaño de ventana, 'q' salir.")

    historiales = {}      # id -> deque[(t, cx, cy, x1, y1, x2, y2)]
    ultimo_visto = {}     # id -> t
    estado = {}           # id -> (moviendose_lateral, moviendose_profundidad)
    mostrar_quietos = False
    fps, t_anterior_real = 0.0, time.perf_counter()

    resultados = modelo.track(source=fuente, conf=args.conf, tracker=args.tracker,
                              imgsz=imgsz, persist=True, stream=True, verbose=False)

    for n_frame, resultado in enumerate(resultados):
        t_real = time.perf_counter()
        fps_inst = 1.0 / max(t_real - t_anterior_real, 1e-6)
        t_anterior_real = t_real
        fps = fps_inst if fps == 0 else 0.9 * fps + 0.1 * fps_inst
        t = n_frame / fps_video if fps_video else t_real

        # El análisis usa la imagen original; solo la de pantalla se redimensiona
        # y las coordenadas se escalan al dibujar.
        alto_img, ancho_img = resultado.orig_img.shape[:2]
        interp = cv2.INTER_AREA if vista < 1 else cv2.INTER_LINEAR
        pantalla = cv2.resize(resultado.orig_img, None, fx=vista, fy=vista,
                              interpolation=interp)

        def P(x, y):
            """Coordenadas de la imagen original -> coordenadas de pantalla."""
            return int(x * vista), int(y * vista)

        cajas = resultado.boxes
        n_moviles = 0

        if cajas.id is not None:
            ids = cajas.id.int().tolist()
            clases = cajas.cls.int().tolist()
            todas_esquinas = cajas.xyxy.tolist()

            for tid, cls, (x1, y1, x2, y2) in zip(ids, clases, todas_esquinas):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                nombre = modelo.names[cls]
                ultimo_visto[tid] = t
                sx1, sy1 = P(x1, y1)
                sx2, sy2 = P(x2, y2)
                linea_1 = (sx1, sy2 + 22)               # posiciones de texto
                linea_2 = (sx1, sy2 + 46)

                def dibujar_inactivo(motivo):
                    if mostrar_quietos:
                        cv2.rectangle(pantalla, (sx1, sy1), (sx2, sy2), GRIS, 1)
                        etiqueta(pantalla, f"{tid} {nombre}", (sx1 + 4, sy1 - 6),
                                 GRIS, 0.55)
                        etiqueta(pantalla, motivo, linea_1, GRIS)

                # Si la caja toca el borde, el objeto está entrando o saliendo:
                # la caja se recorta y no se puede medir su movimiento.
                # if (x1 <= MARGEN_BORDE or y1 <= MARGEN_BORDE or
                #         x2 >= ancho_img - MARGEN_BORDE or y2 >= alto_img - MARGEN_BORDE):
                #     historiales.pop(tid, None)
                #     estado.pop(tid, None)
                #     dibujar_inactivo("en el borde")
                #     continue

                hist = historiales.setdefault(tid, deque(maxlen=LARGO_ESTELA))
                hist.append((t, cx, cy, x1, y1, x2, y2))

                mov = estimar_movimiento(hist, t)
                if mov is None:
                    dibujar_inactivo("midiendo...")
                    continue
                vx, vy, r, retraso = mov
                rapidez = math.hypot(vx, vy)

                # Estados con histéresis
                lat_antes, prof_antes = estado.get(tid, (False, False))
                lateral = rapidez >= (UMBRAL_QUIETO if lat_antes else UMBRAL_MOVIL)
                profundidad = abs(r) >= (UMBRAL_PROF_QUIETO if prof_antes
                                         else UMBRAL_PROF_MOVIL)
                estado[tid] = (lateral, profundidad)

                if not (lateral or profundidad):
                    dibujar_inactivo("quieto")
                    continue

                # ---- Objeto en movimiento ----
                n_moviles += 1
                color = color_para_id(tid)
                # Tiempo hasta el contacto, corregido por el retraso de la medida
                ttc = max(1.0 / r - retraso, 0.0) if (profundidad and r > 0) else None
                alerta = ttc is not None and ttc < TTC_ALERTA_S
                color_caja = ROJO if alerta else color

                cv2.rectangle(pantalla, (sx1, sy1), (sx2, sy2), color_caja,
                              4 if alerta else 2)
                etiqueta(pantalla, f"{tid} {nombre}", (sx1 + 4, sy1 - 6),
                         color_caja, 0.55)
                pts = np.array([P(m[1], m[2]) for m in hist], np.int32)
                if len(pts) > 1:
                    cv2.polylines(pantalla, [pts], False, color, 2)
                cv2.circle(pantalla, P(cx, cy), 4, color, -1)

                textos, consola = [], []
                if lateral:
                    # Flecha: posición prevista dentro de VENTANA_S segundos
                    largo = min(rapidez * VENTANA_S, 200)
                    fin = P(cx + vx / rapidez * largo, cy + vy / rapidez * largo)
                    flecha(pantalla, P(cx, cy), fin, color)
                    # Ángulo: 0° = derecha, 90° = arriba (la y de la imagen va hacia abajo)
                    angulo = round(math.degrees(math.atan2(-vy, vx))) % 360
                    valor = rapidez / args.escala if args.escala else rapidez
                    formato = f"{valor:.2f}" if args.escala else f"{valor:.0f}"
                    textos.append((f"{formato} {unidad}  {angulo} grados", color))
                    consola.append(f"lateral={formato} {unidad} dir={angulo}°")

                if profundidad:
                    flechas_esquinas(pantalla, sx1, sy1, sx2, sy2, r > 0, color_caja)
                    if r > 0:
                        txt = f"se acerca {r:.0%}/s  contacto en {ttc:.1f} s"
                        textos.append((txt + ("  !CERCA!" if alerta else ""),
                                       ROJO if alerta else color))
                        consola.append(f"se acerca {r:.0%}/s TTC={ttc:.1f}s")
                    else:
                        textos.append((f"se aleja {-r:.0%}/s", color))
                        consola.append(f"se aleja {-r:.0%}/s")

                for (texto, col), pos in zip(textos, (linea_1, linea_2)):
                    etiqueta(pantalla, texto, pos, col)
                print(f"ID {tid:<3} {nombre:<12} " + "  ".join(consola))

        # Olvidar objetos que llevan tiempo sin aparecer
        for tid in [i for i, tv in ultimo_visto.items() if t - tv > OLVIDAR_S]:
            historiales.pop(tid, None)
            ultimo_visto.pop(tid, None)
            estado.pop(tid, None)

        # Barra de estado (tamaño de letra fijo, independiente de la vista)
        modo = "todos" if mostrar_quietos else "solo en movimiento"
        etiqueta(pantalla, f"Mostrando: {modo} ({n_moviles} en mov.)",
                 (10, 30), (0, 255, 255), 0.6)
        etiqueta(pantalla, f"m: quietos  +/-: vista {vista:.0%}  proceso: {imgsz} px  q: salir",
                 (10, 60), (0, 255, 255), 0.6)
        texto_fps = f"FPS: {fps:.1f}"
        (ancho, _), _ = cv2.getTextSize(texto_fps, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        etiqueta(pantalla, texto_fps, (pantalla.shape[1] - ancho - 15, 30),
                 (0, 255, 0), 0.7)

        cv2.imshow("Seguimiento YOLO", pantalla)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("m"):
            mostrar_quietos = not mostrar_quietos
            print(f"Objetos quietos: {'visibles' if mostrar_quietos else 'ocultos'}")
        if tecla in (ord("+"), ord("=")):
            vista = min(vista + VISTA_PASO, VISTA_MAX)
            print(f"Vista: {vista:.0%}")
        if tecla in (ord("-"), ord("_")):
            vista = max(vista - VISTA_PASO, VISTA_MIN)
            print(f"Vista: {vista:.0%}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
