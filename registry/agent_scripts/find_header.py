#!/usr/bin/env python3
"""Find a header basename under a project root (args: root name)."""
import os, sys
root, name = sys.argv[1], sys.argv[2]
for dp, _, fns in os.walk(root):
    if name in fns:
        print(os.path.join(dp, name))
