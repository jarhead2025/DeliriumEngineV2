// MOVED: the area-weighted accumulation now lives in kernels/psf_splat.cl
// (kernels `psf_splat_area` and `area_resolve`).
//
// This file was an earlier scratch copy. It is intentionally EMPTY -- the
// renderer builds ONE program from psf_splat.cl, so a second file with
// duplicate helper definitions would be a source of drift.
