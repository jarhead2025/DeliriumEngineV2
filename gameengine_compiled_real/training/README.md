# Entrenamiento del NeuralReconstructor (Intel Arc A770, Linux Mint)

## Instalación (torch >= 2.5: XPU nativo, NO instalar IPEX)

`intel-extension-for-pytorch` esta clavado a versiones exactas de torch
(p.ej. exige 2.8.x) y desde torch 2.5 es innecesario: el backend XPU
para Arc es nativo. Con torch 2.12 estas mejor que con IPEX.

```bash
pip uninstall -y intel-extension-for-pytorch   # si llego a instalarse
# torch ya instalado (2.12); si fuera build CPU-only, reinstalar asi:
# pip install torch --index-url https://download.pytorch.org/whl/xpu
python -c "import torch; print(torch.__version__, '| XPU:', torch.xpu.is_available())"
```

Si `XPU: False`: faltan drivers, no torch. En Mint:
`intel-opencl-icd` + `intel-level-zero-gpu` (repo apt oficial de Intel
para Ubuntu 22.04/24.04 funciona en Mint 21/22), y el usuario en el
grupo `render` (`sudo usermod -aG render $USER`, relogin).

## Uso
```bash
python3 train_reconstructor.py --smoke     # sanidad en CPU, ~1 min
python3 train_reconstructor.py             # A770: ~40 epochs
```

## Qué aprende (etapa 1, alcance honesto)
El AJUSTADOR de parches: ventana 17x17 de muestras de carta (altura +
máscara + color) -> los 8 coeficientes + 12 K de color que el encoder
LSQ clásico produciría (teacher-student / destilación). Etapas futuras:
(2) predecir el layout del quadtree, (3) reconstruir desde fotos. La
salida son los grados de libertad nativos del motor, así que cualquier
etapa hereda render, LOD, costuras y física sin tocar el engine.

## Resultados validados (etapa 1, Intel Arc A770)

- Entrenamiento: 40 epochs (~10 s/epoch), val MAE geom 0.2786 -> 0.0178
  (~16x sobre linea base), colorK 0.0101. Convergencia limpia.
- Puente sobre ventanas REALES del asteroide (97 ventanas):
  residuo medio LSQ 0.2334% vs NEURAL 0.2623% del diagonal
  -> ratio 1.12x, p95 1.03 vs 0.96. STAGE-1 PASS (barra: <1.5x).
- Honestidad de velocidad: a 97 ventanas el LSQ gana (21 ms vs 219 ms;
  el camino XPU paga transferencia + warmup). El cruce favorece a la
  red recien a miles de ventanas por lote. Para objetos chicos, LSQ
  sigue siendo el ajustador correcto; la red apunta a escenas enteras
  y a la etapa 2 (prediccion de layout), donde no existe equivalente
  clasico barato.
