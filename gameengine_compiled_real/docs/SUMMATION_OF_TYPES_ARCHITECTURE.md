# Arquitectura unificada: superficie = Σ Kᵢ·(tipo base), en capas tipo árbol

## La idea del usuario (confirmada y medida)
TODO —geometría, deformación, color— es el MISMO mecanismo: una suma ponderada
de TIPOS base. Y las capas se apilan como un árbol: cada capa es otra suma de
tipos que modifica la anterior. El árbol evalúa hacia abajo hasta un POINT
CLOUD denso (cada punto: posición + propiedades + color) que escupe los píxeles.

    superficie(u,v) = Σ Kᵢ · TipoBaseᵢ(u,v)        [capa 1: forma]
                    + Σ Kⱼ · TipoDetalleⱼ(u,v)      [capa 2: relieve]
    color(u,v)      = Σ Kₖ · TipoColorₖ(u,v)        [capa 3: color]

Una superficie NO es anclas+desplazamiento (lo que hay hoy). Es suma de tipos.

## El error que esto corrige
- El blob que renderizó el usuario era el SpectralPrimitive (campo radial único)
  -> nunca reconstruye la figura. Descartado como geometría.
- El formato de parches actual tiene geometría = Bilinear(anclas)+altura·normal,
  que NO es suma de tipos. Sólo el COLOR ya es suma de tipos (Σ Kᵢ Patternᵢ).
- El rediseño: la GEOMETRÍA también pasa a ser suma de tipos, como el color.

## Las capas (el árbol), cada una suma de tipos — TODO MEDIDO
### Capa 1 — FORMA BASE: tipos de superficie de baja frecuencia
plano(1,u,v), cúpula, silla, valle_u, valle_v, onda_u, onda_v, eggbox, twist...
Medido: parche de esfera RMS 0.6%, pata (cilindro) 2.7%, oreja (silla+pico)
5.2%. ~10-14 tipos reconstruyen parches reales de bajo detalle.

### Capa 2 — RELIEVE/DETALLE: tipos 2D Fourier (sin+cos) de alta frecuencia
Sobre el RESIDUO de la capa 1. Medido: el parche 'rugoso' que la base sola
fallaba (17%) -> con capa de detalle Fourier 2D baja a 1.4%. La capa de detalle
captura lo fino que la base no.

### Capa 3 — COLOR: tipos 2D coseno (ya implementado, PatchColorField)
color(u,v) = Σ Kᵢⱼ cos(πiu)cos(πjv). Captura cualquier textura. Con LOD por
frecuencia (orden bajo lejos, alto cerca).

## Regla de cada capa
Cada capa es Σ Kᵢ·tipoᵢ. Los tipos viven en el ENGINE (costo cero en archivo);
el parche guarda sólo los índices+K. La base de tipos de cada capa se elige
para la frecuencia que esa capa captura (baja=forma, alta=detalle, color=coseno).

## El point cloud y los píxeles
El render evalúa el árbol sobre una grilla (u,v) densa por parche -> puntos 3D
con su color -> se proyectan/splattean a píxeles (el renderer que ya existe,
atom_min de profundidad, Fibonacci no hace falta acá porque el dominio es u,v).
Densidad de la grilla = LOD: más puntos cerca, menos lejos.

## Costuras (la figura se amalgama en los bordes)
Los parches comparten anclas/aristas (sistema de los hitos 1-27). Las superficies
de tipos deben coincidir en las aristas compartidas -> la capa de costura
(stitch s(t), o anclas compartidas como en CageSurface) garantiza gap 0.
Esto se mantiene: la suma de tipos respeta la condición de borde compartida.

## Plan de construcción (piezas)
1. SurfaceTypeLibrary: la base de tipos de FORMA (capa 1) en el engine, con
   evaluador C++ y kernel sincronizados. [empezar acá]
2. Encoder: ajustar Σ Kᵢ·tipo a cada parche de una malla (greedy/LSQ), guardar
   índices+K. Capa de detalle sobre el residuo.
3. Formato: el record del parche guarda índices+K de cada capa (o bloque aparte
   si no entra en 64 bytes).
4. Render: evaluar el árbol por parche -> point cloud -> píxeles. Costuras.
5. Color: ya está (PatchColorField), se integra como capa 3.

## Lo que se rescata
PatchColorField (capa 3 lista), la matemática estable, el renderer de
profundidad, el sistema de anclas/costuras de los hitos 1-27 (la amalgama).

## Análisis del árbol de render (confirmado) y encoder (pieza 2, hecho)
El árbol por parche, en orden:
  NODO 1 FORMA  position(u,v) = sum_i K_i·type_i(u,v)  [K_i vectorial 3D, Opcion B]
  NODO 2 RELIEVE position += sum_j Kd_j·detail_j(u,v)  [Fourier 2D sobre residuo]
  NODO 3 NORMAL  normal = cross(dP/du, dP/dv)          [tras deformar]
  NODO 4 COLOR   color(u,v) = sum_k Kc_k·colortype_k   [PatchColorField]
  HOJA  punto {pos,normal,color} -> splat a pixel (atom_min)

Decision del usuario: NODO 1 = Opcion B (suma vectorial de tipos, SIN anclas).
Amalgama de bordes = MEZCLA de 3: (1) coefs de borde compartidos [primario],
(2) anclas compartidas [refuerzo], (3) capa de costura s(t) [gap residual].

MEDIDO el encoder (encode_patches.py):
- un parche esfera: base RMS 2.15% -> +detalle 0.24% (mejora 89%).
- Spot completo con proyeccion esferica: 2.80% (mala particion, aplasta patas).
- Spot completo con UVs REALES: 0.73% del diagonal (la particion que respeta
  la topologia -> patas/cabeza/cuerpo desplegados, cada parche coherente).
LECCION: el modelo de tipos reconstruye fiel; la PARTICION importa tanto como
el ajuste. Usar las UVs del modelo (o segmentacion que respete la superficie),
NO proyeccion esferica (colapsa las partes salientes).

## Pipeline integrado + amalgama + color real (hito 45-46) - corroborado
PIPELINE end-to-end funciona: encoder->.dlt->fisica->cull->reconstruccion.
.dlt de Spot = 28968 bytes (11x mas chico que OBJ 330624).
CORROBORADO el cuello de botella del RENDER con mediciones:
- 753 vertices de borde compartidos entre parches; cada parche bordea ~6.
- Ajuste independiente -> los parches DERIVAN en los bordes -> nube dispersa.
- AMALGAMA aplicada (peso alto en vertices de borde): gap medio 0.46%->0.29%
  del diagonal, max 0.136->0.061. Los parches convergen a coincidir.
- COLOR real horneado por parche desde la textura via UVs (manchas reales).
LIMITE TECNICO confirmado: el splat de PUNTOS crudo no da superficie solida
aunque el modelo reconstruya bien (0.29% gap, color real). Falta el render
con: densidad adaptativa por parche + normales suaves (cross de la grilla) +
relleno de superficie (triangulacion local del grid u,v por parche, no puntos
sueltos). El modelo de datos esta COMPLETO y correcto; el renderer de puntos
es el componente a reemplazar por un rasterizador de parches (el maduro de
hitos 1-27, o triangular el grid de cada parche).
