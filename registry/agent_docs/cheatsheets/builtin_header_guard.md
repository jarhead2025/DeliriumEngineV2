---
agent: pipeline
topic: header_guard
updated: 2026-09-07T01:03:01+00:00
---

# header_guard

```cpp
// Minimal header layout
#ifndef FOO_H
#define FOO_H
class Foo {
public:
    Foo() = default;
    void bar();
};
#endif // FOO_H
```
