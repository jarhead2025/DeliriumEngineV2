# Analisis arquitectonico: anclas como base vs consecuencia de la superficie

## La observacion del usuario
"usar anchors que en si dependen de una coordenada XYZ sobre la cual
generarse, como si fueran puntos sobre los cuales el resto de la superficie
no se genera." -> las anclas deberian ser CONSECUENCIA de la superficie, no
el cimiento independiente que la deformacion puede contradecir.

## Lo que la medicion mostro (de principio a fin)
1. Posiciones XYZ de anclas: SANAS. 0 fuera del bbox, 0 truncadas.
2. Base bilineal entre anclas compartidas: IDENTICA por construccion en
   bordes compartidos -> no es fuente de desajuste.
3. Coeficientes de deformacion en bordes index-shared: ya armonizados
   (hito 21) -> gap de borde 0.000%.
4. NORMALES de ancla (Nd, la direccion de desplazamiento): AQUI esta el
   problema que el usuario intuye. La superficie se genera como
   P = bilineal + h*Nd. Si Nd contradice la normal geometrica real, la
   deformacion empuja DE COSTADO -> el punto "aparece donde no deberia".
   Medido: 41 anclas (1.1%) con Nd a >30deg de la normal verdadera, hasta
   90deg. ESTO ESTA EN LA GENERACION, como dijo el usuario.

## El fix parcial (PSFBody::RederiveNormals)
Re-deriva las normales de ancla desde la superficie real Y re-ajusta los 8
coefs de cada parche para preservar la geometria. Mejora: 41 -> 12 anclas
torcidas en una pasada.

## El LIMITE de raiz (honesto)
Iterar RederiveNormals NO converge (12->17->13->14, residual oscila ~1.3%).
Causa estructural: el desplazamiento es ESCALAR (h a lo largo de UNA normal).
Cuando la superficie real se aleja LATERALMENTE de la base bilineal, ninguna
magnitud de desplazamiento escalar la alcanza si la normal apunta mal -> hay
un techo. Quedan ~6 anclas reales irreducibles.

El usuario tiene razon a nivel arquitectonico: el modelo "desplazamiento
escalar sobre normal de ancla" tiene un limite intrinseco. El fix COMPLETO
seria desplazamiento VECTORIAL (3 componentes por punto de control en vez de
1 escalar), que captura movimiento lateral. Eso es un cambio de FORMATO
(.psf v4): SurfaceRecord pasaria de 8 floats escalares a 8 vec3, +64 bytes/
parche. Beneficio: representacion exacta sin dependencia de la normal.
Costo: 2x tamano de archivo, re-trabajo de split/merge/modal/stitch.

## Decision
RederiveNormals queda como herramienta OPCIONAL (no por defecto: su residual
de 1.3% empeoraria la geometria de assets ya sanos). El cambio a
desplazamiento vectorial se documenta como la evolucion de formato que
resolveria de raiz la observacion del usuario, para una decision deliberada
(es el cambio mas grande del proyecto y afecta todo el pipeline).
