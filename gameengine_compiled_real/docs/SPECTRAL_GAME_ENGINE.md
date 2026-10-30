# Primitivos espectrales como motor grafico 3D para juegos

## Por que es una representacion de MOTOR, no solo compresion
- ANIMACION GRATIS: animar = mover/rotar/escalar el primitivo. El detalle
  (los coeficientes) viaja con la transformacion, NO se recalcula. Medido:
  trasladar +3 desplaza la superficie +3 exacto, costo cero de detalle.
  No hay skinning por vertice.
- LOD CONTINUO DE UN ARCHIVO: la distancia elige cuantos terminos SH sumar.
  Un .spf de 6KB ES toda la escalera de LOD. Medido: orden 2 (lejos, 0.83%),
  4 (lejos), 8 (medio, 0.79%), 16 (cerca, 0.46%), 32 (close-up, 0.15%). Sin
  almacenamiento extra, sin popping (es continuo).
- COMPOSICION: objeto complejo = union de primitivos (la idea del usuario).
  Una criatura de 3 partes = 19KB; un personaje de ~15 partes ~95KB con
  detalle completo, vs MBs de malla. La fisica mueve las partes; el detalle
  las sigue.

## Presupuesto de frame medido (A770, 60fps, 30% util = 100 GFLOP/frame)
Con LOD por truncamiento:
  orden 32 (close-up): ~130 personajes/frame
  orden 8  (medio):    ~630 personajes/frame
  orden 4  (lejos):    ~2000 personajes/frame
Una escena real mezcla distancias -> miles de objetos viables.

## Subsistema independiente
SpectralPrimitive + SpectralObject + SpectralModel NO dependen del motor de
parches (PSFBody/seams/physics). Es un formato autonomo y mas limpio:
primitivo + campo espectral, transformable, componible, con LOD.

## Camino restante para jugarlo
1. Wiring del kernel spf_splat.cl al renderer (subir coefs, despachar por
   primitivo, una pasada de profundidad compartida) -> correr en A770.
2. Baker desde mallas arbitrarias por particion en partes estrella
   (auto-segmentar un personaje en capsulas + campo por parte).
3. Fisica sobre los primitivos (colision por primitivo, ya hay analitica
   esfera/capsula; el detalle no participa de la colision gruesa).
4. Culling por objeto (frustum + oclusion) antes de despachar samples.
