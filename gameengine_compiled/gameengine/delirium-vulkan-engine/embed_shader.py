#!/usr/bin/env python3
"""Convert a SPIR-V binary to a C++ header + source with an embedded uint32_t array."""
import sys
import os

spv_path = sys.argv[1]
h_path = sys.argv[2]
c_path = sys.argv[3]
name = sys.argv[4]

with open(spv_path, 'rb') as f:
    data = f.read()

# Header
with open(h_path, 'w') as f:
    f.write('#pragma once\n')
    f.write('namespace shaders {\n')
    f.write(f'    extern const uint32_t {name}_data[];\n')
    f.write(f'    extern const size_t {name}_size;\n')
    f.write('}\n')

# Source
with open(c_path, 'w') as f:
    f.write(f'#include "{name}.h"\n')
    f.write('#include <cstdint>\n')
    f.write('#include <cstddef>\n')
    f.write('namespace shaders {\n')
    f.write(f'    const uint32_t {name}_data[] = {{\n')
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        parts = []
        for b in chunk:
            parts.append(f'0x{b:02x}u')
        f.write('        ' + ', '.join(parts) + ',\n')
    f.write('    };\n')
    f.write(f'    const size_t {name}_size = sizeof({name}_data);\n')
    f.write('}\n')

print(f"  Embedded {name}: {len(data)} bytes ({len(data)//4} uint32s)")