# Sobre este proyecto

Actividad en grupo para la unidad 5 de sistemas de percepción

Miembros:

- Toni
- Gerardo
- Juan

# Entregas

- **PDF**, es el fichero llamado `actividad.pdf`
- **Presentción**, el fichero llamado `presentacion.pdf`
- **Código**, el fichero llamado `seguimiento_yolo_final.py`, con instrucciones de como ejecutar más abajo en este documento

El resultado de los videos procesados se pude ver en la carpeta `videos_procesados`

# Instrucciones para ejecutar seguimiento_yolo_final.py

Instalar, numpy, opencv y ultralytics, definidos en `requirements.txt`

Para probarlo con la cámara web:

```bash
python3 seguimiento_yolo_final.py --fuente 0 --modelo yolo26n.pt --reduccion 0.5
```

Para probar el video del almacén:

```bash
python3 seguimiento_yolo_final.py --fuente almacen.mp4 --modelo yolo26n.pt --reduccion 0.5 --vista 0.5
```

Con la tecla m aparecen todos los objetos detectados: quietos y en movimiento
Con - y + reduce el tamaño de la ventana.
