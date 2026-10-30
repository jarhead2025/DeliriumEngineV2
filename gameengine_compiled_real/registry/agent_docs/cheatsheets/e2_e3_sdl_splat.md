# E2-E3 SDL + Splat + OpenCL

## Do NOT touch
- PSFFile::Load, PSFBody::FromFile, FixedReconstructor
- GameObject body=maker.Reconstruct, Scene::objects

## E2
- Optional SDL2 via CMake DELIRIUM_HAS_SDL
- Headless still valid without libsdl2

## E3
- SplatRenderer::Initialize / RenderBody / Present
- OpenCL kernels under kernels/
