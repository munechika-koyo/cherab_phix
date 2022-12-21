# to be run from within manylinux docker container
# run the command below from the root of the source folder
# sudo docker run -ti -v $(pwd):/io quay.io/pypa/manylinux_2_28_x86_64 /bin/bash
# in the container run scripts/build_wheels.sh

PLAT=manylinux_2_28_x86_64

cd /io || exit

# python 3.8
/opt/python/cp38-cp38/bin/python -m pip install build
/opt/python/cp38-cp38/bin/python -m build
auditwheel repair dist/*-cp38-cp38-linux_x86_64.whl --plat $PLAT

# python 3.9
/opt/python/cp39-cp39/bin/python -m pip install build
/opt/python/cp39-cp39/bin/python -m build
auditwheel repair dist/*-cp39-cp39-linux_x86_64.whl --plat $PLAT
