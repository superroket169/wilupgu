#!/usr/bin/env python3
"""Syntax-check CUDA kernel sources embedded in Rust files, without nvcc.

Extracts every r#"..."# raw string, wraps each in its own namespace (some
kernels re-declare helper structs), prepends stubs for CUDA builtins and
compiles with `g++ -fsyntax-only -fno-builtin`.
"""
import re
import subprocess
import sys
import tempfile
import os

PRELUDE = r"""
struct __cu_dim3 { unsigned int x, y, z; };
static __cu_dim3 blockIdx, blockDim, threadIdx, gridDim;
#define __global__
#define __shared__ static
static inline float expf(float x) { return x; }
static inline float logf(float x) { return x; }
static inline float sqrtf(float x) { return x; }
static inline float powf(float x, float y) { return x + y; }
static inline float cosf(float x) { return x; }
static inline float sinf(float x) { return x; }
static inline float fmaxf(float a, float b) { return a > b ? a : b; }
static inline float fminf(float a, float b) { return a < b ? a : b; }
static inline float __uint_as_float(unsigned int u) { return (float)u; }
static inline unsigned int __float_as_uint(float f) { return (unsigned int)f; }
static inline void __syncthreads() {}
static inline float atomicAdd(float* p, float v) { *p += v; return v; }
"""

def check(path: str) -> bool:
    src = open(path).read()
    blocks = re.findall(r'r#"(.*?)"#', src, re.DOTALL)
    if not blocks:
        print(f"{path}: no kernel strings found!?")
        return False
    body = PRELUDE
    for i, b in enumerate(blocks):
        body += f"\nnamespace k{i} {{\n{b}\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(body)
        tmp = f.name
    r = subprocess.run(
        ["g++", "-fsyntax-only", "-fno-builtin", "-Wall", "-Wno-unused-variable", tmp],
        capture_output=True, text=True,
    )
    os.unlink(tmp)
    status = "OK" if r.returncode == 0 else "FAIL"
    print(f"{path}: {len(blocks)} kernels -> {status}")
    if r.returncode != 0:
        print(r.stderr[:4000])
    elif r.stderr.strip():
        print("warnings:", r.stderr[:2000])
    return r.returncode == 0

ok = all([check(p) for p in sys.argv[1:]])
sys.exit(0 if ok else 1)
