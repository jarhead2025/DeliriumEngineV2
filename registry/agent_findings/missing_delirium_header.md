# missing_delirium_header

- agent: pipeline
- tags: include, delirium, psf
- symptom: fatal error: PatternLibrary.h: No such file or directory
- fix: #include "delirium/psf/PatternLibrary.h" with -I engine/include

When g++ reports missing `Foo.h` but `engine/include/delirium/**/Foo.h` exists,
rewrite to `#include "delirium/.../Foo.h"` and compile with `-I engine/include`.
Do not invent a new Foo.h under src/.

```cpp
#include "delirium/psf/PatternLibrary.h"
```

