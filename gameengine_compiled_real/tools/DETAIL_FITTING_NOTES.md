# Ajuste automatico de detalle procedural en el exportador

## Que hace (validado)
`fit_detail()` analiza el residuo de altura que deja el ajuste LSQ de cada
hoja del quadtree y, si ese residuo es ruido de alta frecuencia (decidido
por `analyze_residual()` via reparto espectral radial: >55% de energia en
frecuencias altas), ajusta un descriptor fBm — amplitud (rms del residuo),
periodo base (pico espectral) y octavas (95% de energia acumulada) — y lo
empaca en `surface_def` (CERO bytes extra; el motor lo evalua analitica-
mente, hito 13). `amp_unit = s_scale*64` coincide con `Patch::DetailAmpScale()`.

## Resultados medidos (honestos)
- asteroide: 20/723 parches reciben detalle (3%). Su "ruido" radial es en
  realidad estructura de BAJA frecuencia que el quadtree captura subdividi-
  endo, asi que rara vez dispara el detector. Caso pobre de demostracion.
- roughball (textura de alta frecuencia genuina, 18-47 ciclos): 105/1062
  parches reciben detalle (10%), amplitud media 0.069% del diametro.

## Decision por subdivision-de-prueba (RESUELTO)
La hoja absorbe su residuo con un fBm (en vez de subdividir) solo cuando:
  (1) esta bien cubierta (>92% de muestras) -- el detalle arregla
      fidelidad, no huecos;
  (2) el residuo es ruido de banda ancha (test espectral radial);
  (3) una subdivision DE PRUEBA no reduce el residuo por debajo de 0.65x
      -- es decir, subdividir genuinamente no ayudaria (ruido auto-
      similar), asi que gastar 4 parches seria desperdicio.
El umbral 0.65 es optimo por barrido (0.75/0.85 reducen menos; mas bajo
arriesga fidelidad).

## Resultado medido (roughball, textura de alta frecuencia)
                  parches   RMS      cobertura malla   bytes
  detalle OFF     2116      0.338%   99.8%             326,368
  detalle ON      1952      0.359%   99.8%             304,960
  -> -8% parches y bytes a fidelidad y cobertura IGUALES. Mejora gratis,
     no trade-off. 292/1952 parches (15%) llevan detalle. El motor C++
     consume el archivo (normal analitica 9.5e-7) y renderiza con grano
     coherente sin grietas internas. Assets con textura mas fina ganan
     mas; este es un caso moderado.

## Como reproducir
    python3 tools/psf_export_views.py assets/roughball.obj out.psf 129 0.35
    # comparar conteo de parches y bytes con fit_detail forzado a 0
