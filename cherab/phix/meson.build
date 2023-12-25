py.install_sources(
  [
    '__init__.py',
  ],
  subdir: 'cherab/phix',
)

# Generate version.py for sdist
meson.add_dist_script(
  ['_build_utils/gitversion.py', '--meson-dist', '--write', 'cherab/phix/version.py']
)
if not fs.exists('cherab/phix/version.py')
  generate_version = custom_target(
    'generate-version',
    input: '_build_utils/gitversion.py',
    output: 'version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    build_always_stale: true,
    build_by_default: true,
    install: true,
    install_dir: py.get_install_dir() / 'cherab/phix',
    install_tag: 'python-runtime',
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['version.py'],
    subdir: 'cherab/phix',
  )
endif


subdir('machine')
subdir('observer')
subdir('plasma')
subdir('tools')
