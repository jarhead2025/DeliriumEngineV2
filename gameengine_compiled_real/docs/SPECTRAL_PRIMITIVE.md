# Primitivo espectral: modelos complejos con ecuaciones simples (la vision)

## El objetivo original del usuario
Representar modelos complejos con ECUACIONES simples. Si usamos muchas anclas
para el detalle, reinventamos los vertices/tris -- traiciona el objetivo. El
detalle debe ser un CAMPO matematico horneado sobre un primitivo simple,
igual que una textura 2D hornea detalle sin agregar geometria, pero aca se
hornea detalle 3D integro con menos datos.

## La realizacion (.spf, SpectralPrimitive)
Un objeto = primitivo simple (centro + radio, 16 B) + campo de detalle como
expansion en armonicos esfericos reales:
  superficie(th,ph) = centro + dir(th,ph)*(base_r + Rgeom(th,ph))
  color(th,ph)      = (Rr,Rg,Rb)(th,ph)
  R*(th,ph) = sum c_lm Y_lm(th,ph)
NO hay anclas, NO hay parches, NO hay costuras (un solo campo continuo).

## Medido (validate_spectral, roughball)
- reconstruccion: RMS 0.156% diag = MISMA fidelidad que 3293 parches.
- compresion: 6,412 B vs 464,456 B .psf = 72x mas chico. (vs malla: ~270x)
- continuidad: campo unico C-infinito -> 0 costuras, nada que coser/weldear.
- geom L=32 (1089 coefs), color L=12 (507 coefs).

## Limite y generalizacion (la idea de composicion del usuario)
Armonicos esfericos representan r(th,ph): UN radio por direccion -> sirve
para objetos ESTRELLA (esfera deformada). Toro/personaje NO son estrella.
LA SOLUCION es la composicion que el usuario menciono: objeto complejo =
union de PARTES, cada una un primitivo (esfera/capsula/superquadric) + su
campo espectral. Un personaje = ~15 capsulas x ~1KB = 15KB con detalle
completo. La fisica mueve los primitivos; el detalle viaja con ellos.
"objetos individuales que conforman algo mas grande" = el ensamblaje.

## Camino
SpectralPrimitive evalua forma+color+normal desde el campo. Falta: (1) baker
desde mallas arbitrarias por particion en partes estrella, (2) kernel GPU que
marche direcciones y splatee (trivial, no hay costuras), (3) edicion/fisica
sobre los coeficientes (deformar = modificar pocos numeros). Es el formato
mas comprimido y el que cumple la vision original.
