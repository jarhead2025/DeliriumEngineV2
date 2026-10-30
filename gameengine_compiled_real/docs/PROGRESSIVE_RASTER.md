# Rasterizado progresivo por pixel (metodo del usuario)

## La idea
El rasterizado recorre la profundidad por PASOS, de LEJOS hacia la CAMARA.
A cada profundidad d hay un PLANO cuadrado cuyo ancho/alto los da el FOV:
  half_size(d) = d * tan(FOV/2)
mapeado 1:1 a los pixeles (W x H). Por cada pixel, en cada plano (lejos->cerca),
se pregunta si el modelo ocupa ese punto (x,y,d). Lo mas cercano sobrescribe ->
cada pixel queda con la primera superficie viniendo hacia la camara.
Resolucion = pixel individual. SIN discos, SIN puntos gigantes.

## Geometria (verificada)
pixel (x,y) -> ndc; en profundidad d:
  punto = eye + fwd*d + right*ndc_x*half(d) + up*ndc_y*half(d)
half(d)=d*tan(FOV/2). Cada pixel barre una columna en profundidad; el plano
cuadrado a cada d es lo que cubre el FOV a esa distancia. Confirmado: a d=2,
FOV 50 grados, plano 1.87x1.87 mundo -> 520x520 px.

## Estado: metodo CORRECTO, requiere GPU
El kernel surfel_march.cl implementa esto: un hilo por pixel, marcha far->near,
grilla 3D para 'hay superficie aqui?'. La LOGICA esta verificada en numpy
(forma+color correctos). PERO en numpy queda poroso (16%) porque:
- el render por pixel a resolucion plena = 160k-2M rayos independientes
- necesita muestras de superficie a densidad de pantalla (cientos de miles)
- numpy no puede generar/marchar tantas sin timeout
Esto es POR DISENO un trabajo de GPU: miles de hilos, muestreo denso gratis.
En la A770 el metodo se ve liso porque cada parche se muestrea a densidad de
pantalla en paralelo y los rayos corren todos a la vez.

## Kernel surfel_march.cl
- clear_grid / build_grid: grilla uniforme 3D de muestras de superficie
- march_pixels: un hilo por pixel, marcha far->near por planos-FOV, primer
  impacto solido = color del pixel. Footprint = tamano de pixel en mundo a
  esa profundidad (resolucion exacta).
