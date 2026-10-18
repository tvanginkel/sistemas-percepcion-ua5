# Sobre este proyecto

Actividad en grupo para la unidad 5 de sistemas de percepción

Miembros:

- Toni
- Gerardo
- Juan

# Instrucciones

Instalar, numpy, opencv y ultralytics

Para probarlo con la cámara web:

```pyhton
python3 seguimiento_yolo_final.py --fuente 0 --modelo yolo26n.pt --reduccion 0.5
```

Para probar el video del almacén:

````
python3 seguimiento_yolo_final.py --fuente almacen.mp4 --modelo yolo26n.pt --reduccion 0.5 --vista 0.5```

Con la tecla m aparecen todos los objetos detectados: quietos y en movimiento
Con - y + reduce el tamaño de la ventana.
````
