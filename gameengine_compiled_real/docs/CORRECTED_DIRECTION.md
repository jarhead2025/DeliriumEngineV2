# Rumbo corregido: parches = figura, campo espectral = solo color encima

## La confusión que hubo (y se corrige)
El proyecto se desvió: hice que un ÚNICO campo espectral radial (un radio por
dirección desde un centro) reemplazara la geometría. Eso mató la idea central
y la fidelidad a la figura: Spot salió como un huevo sin patas, porque un campo
radial único NO puede representar una figura real (sólo esferas deformadas).

## El modelo correcto (confirmado por el usuario)
1. **LA GEOMETRÍA = SUPERFICIES (PARCHES) QUE SE AMALGAMAN EN LOS BORDES.**
   Es lo primario. Da la figura real (Spot con patas, asteroide irregular).
   Es el sistema de los hitos 1-27: Patch + AnchorPool + Seam/SeamBridge +
   PSFBody con SeamGraph, costuras a gap 0.0000%. ESTO YA EXISTE Y FUNCIONA.

2. **EL CAMPO ESPECTRAL = SOLO COLOR/TEXTURA, HORNEADO SOBRE LAS SUPERFICIES.**
   NO define forma. NO define relieve. Es la piel de color sobre la geometría
   de parches. Reemplaza/complementa el color actual de los parches
   (EvaluateColor = suma de 16 patrones fijos, muy limitado) por un campo de
   color que puede representar CUALQUIER textura. Acá vive el LOD por
   frecuencia del usuario: la textura se nitidiza con la cercanía.

## Qué se rescata del desvío espectral
- La matemática de armónicos esféricos estable (recurrencia normalizada,
  verificada en C/kernel/baker) -> se reusa para el CAMPO DE COLOR por parche.
- El LOD continuo por banda de frecuencia (sh_field_lod, sin popping) -> se
  reusa para el LOD de la textura de color.
- El muestreo Fibonacci, el atom_min de profundidad -> útiles en el render.

## Qué se deja de lado
- SpectralPrimitive como GEOMETRÍA (el campo radial único para la forma).
  NO representa figuras reales. Queda como curiosidad/caso esfera, no es el
  camino.
- El baker obj->spf que ajustaba GEOMETRÍA a armónicos. El color sí se rescata.

## El trabajo concreto
Hornear un campo de color espectral POR PARCHE (o por región de parches) sobre
la geometría existente:
- Por cada parche, su superficie (u,v) ya está en el mundo. Muestrear el color
  objetivo (textura via UV del modelo) sobre el parche -> ajustar un campo de
  color en armónicos (o en la base 2D del parche) -> guardarlo en el parche.
- El render evalúa el color del parche desde ese campo, con LOD por distancia.
- La FORMA la siguen dando los parches amalgamados, sin tocar.

## Regla de oro (definitiva)
- FIGURA = parches que se amalgaman en bordes. NUNCA un campo radial único.
- COLOR/TEXTURA = campo espectral sobre la superficie, con LOD por frecuencia.
- El campo espectral JAMÁS define la forma.

## Base elegida para el color por parche: 2D COSENO (no armónicos esféricos)
Un parche es un dominio plano (u,v) en [0,1]^2. La base natural es 2D coseno
(tipo DCT), NO armónicos esféricos (esos son para una esfera completa).
Medido sobre una textura de manchas: orden 2 = color grueso (RMS 0.25), orden
8-12 = textura nítida (RMS 0.10). MISMO LOD por frecuencia que el usuario
quería: las frecuencias altas del coseno son el detalle que aparece con la
cercanía. color(u,v) = sum_{i,j} K_ij cos(pi i u) cos(pi j v) por canal.
