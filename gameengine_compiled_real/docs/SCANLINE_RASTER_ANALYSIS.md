# Analisis: rasterizacion por scanline (1 muestra/pixel) -- idea del usuario

## Diagnostico del problema actual (medido)
Los "cortes por encima y por debajo" que el usuario ve NO son:
- geometria solapada: 0 pixeles con dos capas de profundidad (spread max 0.015,
  ninguno supera umbral de pliegue) tras el back-face cull.
- pliegues de profundidad: 0 celdas con parches a distinta profundidad.
SON saltos de SOMBREADO en bordes de parche: 1998 pixeles (0.76%) con salto
de lambert >0.15, concentrados en lineas de parche. Mitigado a 1385 (0.53%)
con promedio ponderado por area + 1 pasada Laplaciana (hito 25).

## La idea del usuario: rasterizar en espacio de pantalla
En vez de que cada parche emita su grilla (u,v) y splatee (densidad heuristica
-> solape Y huecos), procesar la PANTALLA: una scanline por fila de pixeles,
muestreo horizontal = ancho en pixeles, de atras hacia adelante. Densidad
crece sola al acercarse (mas pixeles cubiertos).

## Beneficio real (medido)
- consultas a superficie: ~172,800 (1 por pixel cubierto) vs ~4.4M actuales
  = 25x menos.
- muestras/pixel: exactamente 1.0 -> SIN solape, SIN huecos por construccion.
  Los dos defectos (flaps y gaps) desaparecen estructuralmente, no por tuning.
- trabajo: ~124 MFLOP (raycast con binning) vs ~132 MFLOP actual: comparable.

## Costo / riesgo (por que es rediseno, no ajuste)
Cada pixel necesita saber QUE parche proyecta ahi y su (u,v). Inversion
screen->uv. Dos caminos:
 (a) ray cast por pixel: el motor YA tiene Raycast/Newton (hito 6). Con binning
     de parches por bbox de pantalla (~12 parches/pixel), ~2.07M tests
     ray-parche. Riesgo: Newton puede no converger en parches muy deformados.
 (b) scanline clasico con active-edge-table: mantener parches que cruzan la
     scanline, avanzar fila a fila. Mas complejo de implementar en GPU.

## Recomendacion
Es el enfoque arquitectonicamente correcto y MAS eficiente. Es el proximo
gran paso del renderer. Requiere: (1) binning de parches en tiles de pantalla,
(2) test ray-parche por pixel reusando ClosestPointEx/Raycast del hito 6,
(3) z-min por pixel (ya existe). Hacerlo bien necesita su propia sesion con
gates de convergencia del Newton y cobertura.
