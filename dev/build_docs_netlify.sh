#!/bin/bash

echo Install dependencies...

# install dependencies from github
source requirements/github_pkgs.sh
# for doc
pip install -r requirements/docs.txt

# install cherab_phix

source dev/install.sh

echo Building docs...

cd docs

make html
