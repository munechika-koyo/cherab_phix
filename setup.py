from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "cherab"

THREADS = os.cpu_count() or 1
FORCE = False
PROFILE = False

# === Set arguments ===========================================================
if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]
else:
    force = FORCE

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]
else:
    profile = PROFILE

compilation_includes = [".", numpy.get_include()]
compilation_args: list[str] = []
cython_directives = {"language_level": 3, "profile": profile}

# === Build .pyx extension ====================================================
extensions = []
for pyx in SRC.glob("**/*.pyx"):
    pyx_path = pyx.relative_to(ROOT)
    module = ".".join(pyx_path.with_suffix("").parts)
    extensions.append(
        Extension(
            module,
            [str(pyx_path)],
            include_dirs=compilation_includes,
            extra_compile_args=compilation_args,
        )
    )

# generate .c files from .pyx
extensions = cythonize(
    extensions,
    nthreads=THREADS,
    force=force,
    compiler_directives=cython_directives,
)

# === Define setup function ===================================================
setup(
    ext_modules=extensions,
)
