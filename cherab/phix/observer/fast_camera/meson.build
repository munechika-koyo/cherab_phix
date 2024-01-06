_cython_tree = declare_dependency(sources: [
  fs.copyfile('__init__.pxd'),
  fs.copyfile('colour.pxd'),
])

cython_sources = [
  ['colour', 'colour.pyx'],
]

# compile cython sources
foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: [py_dep, np_dep, _cython_tree],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/phix/observer/fast_camera',
  )
endforeach

python_sources = [
  '__init__.py',
  '__init__.pxd',
  'camera.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/phix/observer/fast_camera',
)

subdir('pipeline')
subdir('data')
subdir('tests')
