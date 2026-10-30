# include_fix_Scene.cpp

- agent: pipeline
- tags: include, bulk, engine
- symptom: missing include in engine/src/core/Scene.cpp
- fix: Scene.h -> "delirium/core/Scene.h"; GameObject.h -> "delirium/core/GameObject.h"

Deterministic include rewrite.

Scene.h -> "delirium/core/Scene.h"
GameObject.h -> "delirium/core/GameObject.h"
