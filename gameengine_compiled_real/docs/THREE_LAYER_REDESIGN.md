# Rediseño: separación geometría / relieve / textura

## El error que corregimos
El baker horneaba la GEOMETRÍA con armónicos esféricos de orden alto (32),
el MISMO mecanismo que el detalle de textura. Eso fue una confusión de capas:

- La forma de un modelo bien comportado vive casi entera en órdenes BAJOS.
  Medido en roughball: **órdenes 0-2 = 99.9% de la energía de forma**. Los
  órdenes 8/16/32 no aportan a la silueta.
- Forzar la geometría a orden 32 sólo servía para que las partes no-estrella
  (patas, hocico de Spot) inyectaran ENERGÍA en las bandas altas (órdenes
  20-31), y la superficie evaluada EXPLOTABA (radios de -17 a +18, rayos
  disparados). El render del usuario mostró exactamente eso.
- Además, variar el orden de geometría con la distancia (LOD) cambiaba la
  SILUETA, no sólo el detalle -> la forma "morpheaba" con el zoom.

## El diseño correcto: TRES capas separadas

### Capa 1 — FORMA BASE (geometría)
Armónicos esféricos de **orden bajo fijo** (4-8). Da la silueta: estable,
suave, sin oscilaciones, IMPOSIBLE que explote (no hay bandas altas que
inyecten energía). NO depende de la distancia. Para Spot da un cuerpo limpio
sin patas (las patas son otra cosa -> composición en partes).

### Capa 2 — RELIEVE FINO (geometría de detalle, SEPARADA)
Las arrugas/bultos finos de la superficie, como una capa aparte de la forma
base. NO son "más órdenes del mismo ajuste" (eso explota). Se construye
después si el modelo lo necesita; para muchos juegos, forma base + textura
alcanza. Pendiente de diseño (posible: mapa de desplazamiento espectral
acotado, o detalle por parche).

### Capa 3 — TEXTURA / COLOR (la idea original del usuario)
Campo espectral de color con **LOD por orden de frecuencia**: la textura se
vuelve nítida con la cercanía (las manchas de Spot emergen al acercarse). Acá
el orden SÍ escala con la distancia, de forma continua, sin popping. Esta capa
ya funciona y se valida con Spot.

## Regla de oro
- GEOMETRÍA: orden bajo fijo. La forma NO cambia con la distancia.
- TEXTURA: orden variable con la distancia. El detalle de color aparece con
  la cercanía.
- Son DOS representaciones distintas. No volver a mezclarlas.

## Qué cambia en el baker
- `--lgeom` default 32 -> **6** (forma base estable).
- `--lcolor` se mantiene alto (12+): la textura sí quiere frecuencia.
- La regularización geométrica deja de ser un parche: con orden 6 la forma no
  puede explotar, no hace falta clampear radios ni penalizar bandas altas.
- El LOD del visor ya NO toca la geometría (orden geom fijo), sólo el color.

## Qué NO resuelve esto (honesto)
Spot seguirá sin patas: la forma base de orden bajo es un cuerpo redondeado.
Las patas requieren DESCOMPOSICIÓN en partes (cuerpo + patas + cabeza, cada
una su primitivo). Eso es el siguiente gran paso, ortogonal a esta separación.
