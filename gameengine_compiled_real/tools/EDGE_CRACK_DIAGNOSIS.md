# Diagnostico: bordes que estallan en assets de alta frecuencia (roughball)

## Sintoma
Al renderizar roughball.psf en el editor, el cuerpo es suave pero los
bordes "estallan" en parches sueltos y hay agujeros negros. El usuario
noto que "algo funciona y en algun momento lo rompe".

## Lo que NO era (descartado con tests)
- NO son las operaciones del editor: la geometria esta corrupta YA en el
  archivo cargado (gap 4.7% antes de tocar nada). dbg_health: 0 indices
  de ancla fuera de rango, 0 parches degenerados tras toda la secuencia.
- NO es el detalle procedural: gap identico con surface_def=0.
- NO es saturacion de cuantizacion del stitch (sola): subir s_scale de
  0.012 a 0.06*diag da rango de sobra, pero FitAll igual no cierra.
- NO es un gap tangencial: la descomposicion da 100% normal, 0% tangencial
  -> el stitch SI podria cerrarlo geometricamente.
- NO es que el stitch no se aplique: escribir un byte mueve la superficie
  exactamente lo esperado (dbg_stitchapply: STITCH WORKS).

## Causa raiz (encontrada)
El exportador genera UNIONES-T MAL FORMADAS entre hojas del quadtree de
distinto nivel. La peor costura (gap 4.46%) tiene a_is_finer=1 y sus
endpoints NO comparten ancla: lado A anclas {1432,1434}, lado B {1426,1432}
-- solo una en comun. El stitch de union-T asume que el lado fino cose
contra el medio del borde grueso con una correspondencia tb0..tb1 limpia,
pero cuando las hojas vecinas diferen en >1 nivel de subdivision esa
correspondencia no existe, y el stitch no puede cerrar el salto.

386 costuras tienen gap >1% diag; casi todas son uniones-T de este tipo.

## Arreglos
1. PARCIAL (hecho): s_scale adaptativo (0.06*diag base) da al stitch el
   rango para las costuras que SI estan bien formadas. El weld armonico
   (Soft) reduce 386 -> 204 grietas grandes moviendo anclas.
2. DE RAIZ (pendiente, es reescritura del exportador): forzar BALANCE 2:1
   entre hojas vecinas del quadtree -- ningun borde puede tener un vecino
   de mas de un nivel de diferencia. Es la MISMA regla que ya funciona en
   PSFBody::MergeGroup (hito 9). El exportador debe, tras construir el
   quadtree, subdividir hojas hasta que todo par de vecinos difiera <=1
   nivel, ANTES de fijar las anclas de borde. Esto elimina las uniones-T
   mal formadas en origen.

## Estado
roughball es un caso extremo (textura de muy alta frecuencia -> quadtree
muy desbalanceado). Assets mas suaves (torus, asteroid) no exhiben esto
(sus tests de costura pasan). El motor es correcto; el exportador necesita
el balance 2:1 para assets de alta frecuencia.
