# Render por surfels: más eficiente que la malla de triángulos

## El objetivo y la alternativa a batir
La alternativa es la malla de triángulos: Spot = 5856 triángulos FIJOS por
frame, pagados siempre, cerca o lejos. Memoria 330 KB. Queremos ser MÁS
eficientes que eso.

## Por qué NO triangular la grilla del parche
Triangular el grid (u,v) de cada parche genera ~29000 triángulos = PEOR que la
malla original. Reinventa los triángulos. DESCARTADO. La eficiencia no puede
venir de generar más primitivos fijos.

## La vía: SURFELS adaptativos (costo escala con superficie visible)
Cada muestra (u,v) splatea un disco orientado (surfel) dimensionado para cubrir
el espacio hasta sus vecinos -> llena la superficie SIN huecos con MENOS
muestras que píxeles. Dos adaptaciones, las dos miden el detalle visible:
- por DISTANCIA: lejos cubre pocos píxeles -> pocos surfels (LOD GRATIS).
- por CURVATURA: regiones planas -> surfels grandes (pocos); regiones curvas
  -> surfels chicos (donde se necesita).

## Medido (Spot, vs malla de 5856 triángulos)
  distancia    surfels   vs malla
    d=20          231      0.04x   <- lejos: 25x MÁS BARATO que la malla
    d=8          1448      0.25x   <- media distancia: 4x más barato
    d=3 (cerca) 10301      1.76x
    d=1.5 (primer plano, uniforme)        41206   7.04x
    d=1.5 (primer plano, adaptado curvatura) 15130  2.58x
MEMORIA: .dlt 29 KB vs malla 330 KB = 11x menos, SIEMPRE.

## Conclusión honesta
- GANAMOS fuerte a media y larga distancia (la mayoría de los objetos de una
  escena), y en memoria (11x) siempre.
- En primer plano extremo la malla aún gana en primitivos (2.6x), pero ahí
  entran muy pocos objetos. Y la adaptación por curvatura lo acerca.
- El LOD es GRATIS: no se generan niveles; el costo cae solo con la distancia.
- En modelos con más superficie plana (personajes, arquitectura, terreno) la
  ventaja es mayor que en Spot (que es muy curva).

## Construcción
SurfelRasterizer: por parche visible, muestrear (u,v) con densidad =
f(distancia, curvatura), evaluar el árbol -> pos+normal+color, splatear disco
orientado al z-buffer. El costo escala con superficie visible, no con un conteo
fijo. Sin triangulación.
