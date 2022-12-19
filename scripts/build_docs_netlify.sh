#!/bin/bash

echo "Download git-lfs contents"
wget https://github.com/git-lfs/git-lfs/releases/download/v3.1.4/git-lfs-linux-amd64-v3.1.4.tar.gz
tar xvfz git-lfs-linux-amd64-v3.1.4.tar.gz
# Modify LFS config paths to point where git-lfs binary was downloaded
git config filter.lfs.process "`pwd`/git-lfs filter-process"
git config filter.lfs.smudge  "`pwd`/git-lfs smudge -- %f"
git config filter.lfs.clean "`pwd`/git-lfs clean -- %f"
# Make LFS available in current repository
./git-lfs install
# Download content from remote
./git-lfs fetch
# Make local files to have the real content on them
./git-lfs checkout

echo "Install cherab-phix"
python -m pip install .[dev,docs]

# echo "Install pandoc"
# apt install pandoc

echo "Build document"
python dev.py doc -j1
