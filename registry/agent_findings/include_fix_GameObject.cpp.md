# include_fix_GameObject.cpp

- agent: pipeline
- tags: include, bulk, engine
- symptom: missing include in engine/src/core/GameObject.cpp
- fix: GameObject.h -> "delirium/core/GameObject.h"; PSFBody.h -> "delirium/psf/PSFBody.h"

Deterministic include rewrite.

GameObject.h -> "delirium/core/GameObject.h"
PSFBody.h -> "delirium/psf/PSFBody.h"
