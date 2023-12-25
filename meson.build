project(
  'cherab-phix',
  'c',
  'cython',
  version: run_command(
    ['cherab/phix/_build_utils/gitversion.py'],
    check: true
  ).stdout().strip(),
  meson_version: '>= 0.64.0',
  default_options: [
    'cython_args=-3',
    'c_std=c99',
    'buildtype=debugoptimized',
  ],
)

fs = import('fs')
py = import('python').find_installation(pure: false)

# Python dependency
py_dep = py.dependency()

# OpenMP dependency
omp_dep = dependency('openmp', required: true)

# NumPy dependency
incdir_numpy = run_command(
  py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

# Numpy deprecated api (if errors is occurred, comment out in module section)
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# subdirectories
subdir('cherab/phix')
