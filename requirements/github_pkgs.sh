# GitHub packages
# To compile them, numpy and cython must be installed in advance
# install raysect development branch
# pip install git+https://github.com/raysect/source.git@development#egg=raysect

# install cherab munechika-koyo remote repo, koyo_dev branch
pip install git+https://github.com/munechika-koyo/core.git@koyo_dev#egg=cherab
# git clone -b koyo_dev https://github.com/munechika-koyo/core.git
# python core/setup.py install build_ext -i

# install calcam
pip install git+https://github.com/euratom-software/calcam.git#egg=calcam
# python calcam/setup.py install