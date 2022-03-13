# install calcam
BASE_PATH=$(pwd)

# clone repo into HOME
cd $HOME
git clone https://github.com/euratom-software/calcam.git
cd calcam
python setup.py install
cd $BASE_PATH