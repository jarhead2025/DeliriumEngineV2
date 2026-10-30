# Compilar y correr en GPU real (hito 3)

Validado hasta donde el sandbox permite: la matemática del kernel está
probada contra el emulador CPU, y todo el host pasa verificación de
sintaxis/firmas. La primera ejecución real es en tu máquina.

## Linux (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake libsdl2-dev ocl-icd-opencl-dev
# driver NVIDIA ya instalado -> provee el ICD de OpenCL
mkdir build && cd build
cmake .. && make -j
ctest                       # compuertas de los hitos 1 y 2
cd .. && ./build/example_torus
```

## Windows (MSYS2/MinGW o Visual Studio)

- SDL2: vcpkg `vcpkg install sdl2 opencl` o el devel pack de libsdl.org
- OpenCL: viene con el driver NVIDIA; el SDK de headers via vcpkg
- `cmake -B build && cmake --build build --config Release`

## Qué esperar

- Título de la ventana: fps, muestras/píxel (~5-6) y nombre del device.
- ESPACIO: impulso modal sobre el toro (debería ondular y amortiguarse
  en ~1 s, sin grietas). ESC: salir.
- En la GTX 1650 el cuadro debería costar <1 ms de GPU; el límite será
  el readback CL->SDL (milestone futuro: interop o doble buffer).

## Si algo falla

1. `kernel build log` impreso -> error de compilación del .cl en tu
   driver: pegalo en un issue/al chat.
2. Imagen negra con fps altos -> revisar que assets/torus.psf existe
   (se genera con `python3 tools/psf_validate.py` o psf_encode.py).
3. Imagen corrupta -> probable divergencia CPU/GPU: correr
   `python3 tools/psf_validate.py assets/torus.psf` y comparar
   psf_render.png contra lo que ves.
