#!/usr/bin/env python3
"""Echo a finding parallel template (args: symptom fix)."""
import sys
sym = sys.argv[1] if len(sys.argv) > 1 else ""
fix = sys.argv[2] if len(sys.argv) > 2 else ""
print(f"PARALLEL\nsymptom: {sym}\nfix: {fix}")
