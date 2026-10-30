# Experimental conclusions retained by the revision

## Retained

### 1. The physical object is not the render cache

The earlier work established that a surfel file can render a complex object at interactive rates. It also established that surfels alone are a poor editable/physical source of truth. The revision retains:

```text
canonical body (PSF/cage) → derived surfel cache → GPU march → pixels
```

A surfel cache may be regenerated, discarded, or partially updated without losing the object’s topology, material meaning, skeleton bindings, or simulation state.

### 2. Articulation precedes surface reproduction

The skeleton produces pose targets. Spring/damper secondary motion may make that target dynamic, as in VTuber-style rigs. The resulting bone state drives anchor targets. The body/constraint solve then determines the final anchor state and surface. This avoids treating a rendered skin as the physical skeleton.

### 3. Shared boundaries are better than cosmetic correction

The scalar PSF implementation uses shared anchors and seam correction. The cage experiment showed that when adjacent patches share the same moving boundary degrees of freedom, positional continuity and derived normals are structurally stronger. The revision keeps scalar PSF compatibility but treats cage/shared-boundary evaluation as the preferred direction for future continuity work.

### 4. GPU work must be bounded and measurable

The progressive marcher is valuable because it maps a finite repeated computation to GPU ALUs: ray sample → spatial candidate query → coverage/depth decision → composition. M0 retains the quality settings as explicit budget parameters: resolution, layers, band steps, fill passes, surfel density, and cache-update work.

### 5. Renderer profiling and renderer throughput differ

An OpenCL profiling path may wait for individual events. This measures device stage cost but reduces pipeline concurrency. Throughput and profiling runs must be stored separately.

## Deferred

### Vector-valued patch residuals

Scalar displacement along an interpolated normal has a lateral-detail ceiling. Vector residuals are a likely future extension, but M0 does not choose the binary layout yet.

### The experimental `.dlt` tree model

The tree/basis experiment is valuable as an offline fitting/reconstruction approach. It is not yet the runtime canonical representation because it is not integrated with the active renderer, physics, seams, or asset path.

### Full material fields

The existing material model is a useful seed. Temperature, electromagnetic, moisture, combustion, and other field simulations remain later additions after articulated mechanical state is stable.

## Rejected as primary architecture

### Independent raw SRF authoring

An independently produced surfel asset can disconnect render shape from physical/edited shape. SRF is retained only as a generated cache with source-body and generation metadata.

### Blind overlap/color averaging

Overlapping surface samples cannot all be summed without considering depth, normals, and material ownership. This causes self-occlusion and material bleeding. Blending is allowed only for compatible samples of the same continuous surface region; nearest valid surface remains the primary depth owner.

### Post-fill as the sole watertightness mechanism

Neighborhood fill can hide sub-pixel gaps. It must not replace correct topology, surfel footprints, or spatial candidate queries.
