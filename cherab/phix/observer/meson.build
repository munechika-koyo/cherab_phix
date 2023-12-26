cython_sources = [
  ['thin_lens_ccd', 'thin_lens_ccd.pyx'],
]

# compile cython sources
foreach file : cython_sources
  py.extension_module(
    file[0],
    file[1],
    # c_args: [numpy_nodepr_api],
    dependencies: [py_dep, np_dep],
    install: true,
    install_dir: py.get_install_dir() / 'cherab/phix/observer',
  )
endforeach

python_sources = [
    '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: 'cherab/phix/observer',
)

subdir('fast_camera')
