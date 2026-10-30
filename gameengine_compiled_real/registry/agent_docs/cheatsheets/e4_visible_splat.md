# E4 visible splat

1. Set device_name from clGetDeviceInfo
2. RenderBody: upload AnchorBlob/SurfaceBlob to CL buffers; write FB
3. Present: clEnqueueReadBuffer then SDL_LockTexture / RenderPresent
Do NOT rewrite PSFFile/PSFBody/FromFile/GameObject/Scene.
Later: load kernels/psf_splat.cl via BuildProgram.
