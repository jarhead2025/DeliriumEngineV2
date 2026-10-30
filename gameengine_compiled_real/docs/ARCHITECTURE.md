# Arquitectura DeliRium v0.2

## Invariantes (no negociables)

1. **Layout binario congelado**: `AnchorRecord`, `SurfaceRecord` y
   `BoneRecord` miden 64 bytes con offsets fijos, verificados por
   `static_assert` y compartidos textualmente entre C++, los kernels y
   las tools Python. Cualquier cambio rompe tres mundos a la vez.
2. **CPU y GPU evalúan la misma matemática**: `Patch::Evaluate` debe ser
   bit-compatible con `psf_splat.cl`. El validador es `tools/psf_validate.py`.
3. **Base que se anula en esquinas**: la superficie interpola sus anclas
   pase lo que pase con los coeficientes → las grietas de esquina son
   imposibles por construcción, y cada arista pertenece a exactamente 2
   coeficientes → coser = ajustar 2 números.
4. **Toda edición es local y pequeña**: tocar un objeto = reescribir
   algunos registros de 64 B (y subir solo esos a VRAM).

## Flujo de un cuadro

```
fixed step:  ModalDynamics::Step (todos)        ← escala barata
             LocalFEM::Step (regiones calientes) ← escala cara
             CollisionAnalytic → impulsos
             SeamGraph: MaxGap() bajo tolerancia
render:      FlushToGPU (solo registros sucios)
             BuildSampleGrids (anisotrópico, ss_u/ss_v)
             clear_fb → psf_splat → resolve_fb
```

## El contrato del reconstructor inteligente

Entrada: descripción semántica + restricciones físicas + presupuesto.
Salida: registros de 64 B (anclas, superficies, huesos) + sugerencias de
modos. Pasa por `ReconstructionModel::Validate` antes de entrar al motor.
Como los parches/K/costuras son los grados de libertad nativos del motor,
un modelo que aprende a emitirlos hereda gratis: render, LOD infinito,
física modal y deformación sin grietas.

## Números de referencia (medidos en los experimentos del repo)

- Compresión a fidelidad igual: 24× (formas curvas) a 117× (terreno suave)
  vs malla binaria; piso ~6× en contenido de alta frecuencia.
- Toro: 544 parches = 72,5 KB, RMS 0,167 % de la diagonal, grieta máxima
  cosida 0,038 %.
- Splat: sin huecos a ~5,7 muestras/píxel; ~115 FLOP/muestra; 0,19 % del
  cómputo de una GTX 1650 en el cuadro de prueba; escritura ~3 % del bus.
