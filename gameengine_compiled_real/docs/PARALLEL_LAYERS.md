# Capas en paralelo: física y render desacoplados (pregunta del usuario)

## La pregunta
¿La deformación debe correr en serie dentro del bucle gráfico, o pueden ir
capas separadas, ya que una es causa-consecuencia de la otra pero no dependen
de encadenarse, sino de un espacio pre-asignado que cada proceso va cambiando?

## La respuesta: SÍ, se paralelizan (la intuición es correcta)
Es el patrón PRODUCTOR-CONSUMIDOR con DOBLE BUFFER.

ESPACIO COMPARTIDO: posiciones de anclas + propiedades por parche (2 buffers).

PROCESO FÍSICA (su ritmo, ej. 120 Hz, hilo o cola-GPU aparte):
  lee anclas -> aplica tensión-deformación / calor / EM -> escribe anclas.
  No espera al render.

PROCESO RENDER (su ritmo, tasa del monitor):
  lee las anclas MÁS RECIENTES -> reconstruye superficies -> rasteriza surfels.
  No espera a la física; usa el último estado disponible.

## Por qué funciona
- Se comunican por el ESPACIO COMPARTIDO, no se llaman entre sí. Desacoplados.
- Causa-consecuencia se mantiene con 1 frame de desfase (~8-16ms),
  imperceptible al ojo.
- Permiten RITMOS DISTINTOS: física estable a 120Hz, render a 60/144Hz.
- Doble buffer (la física escribe en B mientras el render lee A, luego swap)
  evita que se pisen sin bloqueo.

## Escala a las capas físicas futuras
Cada fenómeno (mecánica, calor, EM) es OTRO proceso que lee/escribe el mismo
espacio compartido a su propio ritmo. El calor difunde lento (pocas Hz), la
mecánica rápido. Todos escriben propiedades por ancla/parche; el render lee el
estado actual. "El motor traduce entre comportamientos" = todos comparten el
mismo espacio de estado, cada uno lo modifica, ninguno bloquea a otro.

## La excepción honesta
Si el render del frame N necesita EXACTAMENTE la física del frame N (demo
determinista, o física que reacciona al render mismo), se sincroniza. El caso
común de deformación NO lo necesita -> se paraleliza y gana fps.
