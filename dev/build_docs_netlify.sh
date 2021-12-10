#!/bin/bash

echo Install dependencies...

pip install -r requirements/build.txt
pip install -r requirements/install.txt

# install from github repos
source requirements/github_pkgs.sh

pip install -r requirements/docs.txt

echo Install cherab_phix...

source dev/build.sh
source dev/install.sh

echo Building docs...

cd docs

make html
