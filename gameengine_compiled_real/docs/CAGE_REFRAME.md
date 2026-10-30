# El replanteo de cage: anclas como grados de libertad (idea del usuario)

## La idea
"un replanteo de como la deformacion afecta los anchors en si, y recien ahi
la superficie generada: los anchors generan un volumen primitivo fisico, y
luego las superficies se ajustan a los mismos."

Inversion del modelo: en vez de anclas FIJAS + deformacion escalar que infla
la superficie entre ellas (a lo largo de una normal que puede estar mal), las
ANCLAS son los grados de libertad. La deformacion las MUEVE. La superficie es
lo que se ajusta suavemente a donde quedaron. Las anclas = volumen primitivo;
la superficie = piel sobre el.

## Por que es estructuralmente superior (TODO MEDIDO en validate_cage)
- COSTURAS GRATIS: dos parches que comparten arista comparten sus 2 anclas, y
  la superficie esta determinada por las posiciones de ancla -> la arista
  compartida es UNA curva en ambos lados -> gap 0.0000% SIN weld/stitch/
  harmonize/bridge. Todo el subsistema de costuras SE DISUELVE.
- NORMALES NO PUEDEN CONTRADECIR: la normal se deriva del cage (cross de
  aristas vecinas), es consecuencia de la superficie -> el modo de falla
  "el punto aparece donde no deberia" es IMPOSIBLE.
- DEFORMACION FISICA: mover un ancla mueve la piel; anclas compartidas
  acoplan vecinos automaticamente. (Esto tambien simplifica la fisica modal:
  los modos ya son movimientos de ancla.)

## El costo (medido)
La superficie de cage difiere de la actual (con coefs) en 0.15% RMS -- el
detalle fino que los 8 coefs/parche encodean. Para hacerlo estructural hay
que migrarlo al cage como anclas extra (1 nivel de subdivision donde hay
detalle: ~3709 -> ~5563 anclas en zonas detalladas). Trade-off: +50% anclas
en zonas de detalle, PERO se elimina todo el codigo de costuras y los 0 bytes
de coefs por parche (8 floats = 32 bytes/parche menos).

## Estado
CageSurface implementado y validado (gap 0.0000%, normal consistente). Es un
MODO ALTERNATIVO de evaluacion, lado a lado con el actual, sin romperlo. La
migracion completa (hacer el cage el modelo primario) implica: subdividir
donde haya detalle >umbral, quitar los coefs del formato (.psf v4 cage),
re-derivar split/merge/modal/render sobre el cage. Es el camino que resuelve
de raiz TODOS los problemas de costura y normal que perseguimos -- la
recomendacion es migrar, por etapas, validando cada una.
