# Legacy - codigo de eras anteriores (preservado, no en el build activo)

Este motor evoluciono por eras. El codigo aqui ya NO se usa en el pipeline
actual (era 4: arbol suma-de-tipos + segmentacion tela + surfels + fisica de
anclas + render progresivo por pixel), pero se preserva porque cada era dejo
lecciones medidas.

## era2_spectral/ - campo radial armonico esferico (ABANDONADO como geometria)
Un solo campo radial surface(theta,phi)=centro+dir*(r+sum c_lm Y_lm).
LECCION CLAVE: un campo SH radial unico NO puede representar figuras reales
(Spot -> blob sin patas). La idea de LOD por frecuencia era correcta pero
pertenece al COLOR (ahora en PatchColorField), no a la geometria.
Incluye: SpectralPrimitive, SpectralObject, SpectralRenderer, SplatRenderer,
ejemplos 03/04, tests spectral/harmonic/chartseam.

## era1_psf/ - NOTA
Los modulos PSF originales (Patch, AnchorPool, Seam, PSFBody, CageSurface,
PatternLibrary, ProceduralDetail, Reconstructor) NO se movieron aqui porque
estan entrelazados con la infraestructura del nucleo (Engine, GameObject) y la
fisica (ModalDynamics, LocalFEM). Solo se movio AnchorPool_snapped (experimento
suelto). Limpiar era1 requeriria desacoplar el nucleo primero - trabajo futuro.

## Activo (era 4) en engine/src:
psf/SurfaceTypeLibrary, psf/PatchColorField, psf/TreeFormat, psf/AnchorPhysics,
render/SurfelRasterizer + kernels/surfel_march.cl + tools/march_gpu_render.cpp
