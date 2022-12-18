#!/bin/bash

echo "Install cherab-phix"
python -m pip install .[dev,doc]

echo "Install pandoc"
apt install pandoc

echo "Build document"
python dev.py doc
