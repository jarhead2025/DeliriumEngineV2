# Segmentación "tela": parches = grupos de triángulos que se adaptan

## La idea (del usuario)
Como un pedazo de tela que cae sobre la vaca y se ajusta a su superficie: un
parche es una REGIÓN CONTIGUA de la superficie real, no un cuadrilátero
arbitrario. Y la vaca YA viene como triángulos en su formato original, así que:
- NO inventamos la partición desde cero: AGRUPAMOS triángulos existentes.
- Cada parche (grupo de tris contiguos) se aproxima con la suma de tipos.
- Los tris comparten aristas -> los bordes de parches son CONTINUOS (sin
  agujeros) por construcción.

## El crecimiento "tela"
Desde un triángulo semilla, el parche CRECE agregando triángulos vecinos
mientras la suma-de-tipos siga ajustando bien (error < umbral). Cuando agregar
un tri rompería el ajuste, se detiene -> ese parche está completo, empieza otro.
El parche se adapta a cuánta superficie pueden cubrir los tipos, como la tela
conformándose: zonas suaves = parche grande (muchos tris), zonas con detalle =
parche chico (pocos tris).

## El beneficio (donde ganamos a la malla) - MEDIDO en Spot
- 5856 triángulos -> 59 parches = **99x menos primitivos**.
- Fidelidad: 0.367% del diagonal.
- Zonas suaves absorben hasta 200 tris en 1 parche; zonas detalladas quedan
  chicas. La compresión ocurre justo donde hay tris redundantes.
- Bordes: gap medio 0.274% (los vértices de tri compartidos son el MISMO
  punto -> parches vecinos coinciden ahí; la amalgama lo lleva a ~0).

## Por qué esto resuelve el cuello de botella
La segmentación PCA en cuadriláteros no respetaba la topología -> surfels
sueltos en bordes. La tela usa los bordes REALES de la malla -> continuidad
garantizada, y la compresión es ADAPTATIVA (mucha donde sobran tris, poca donde
hay detalle). "Cuando sean demasiados tris, ahí veremos el beneficio" (usuario).
