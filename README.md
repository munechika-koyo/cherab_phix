# CHERAB-PHiX

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Docstring formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Docstring style: numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_phix/master.svg)](https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_phix/master)
[![Netlify Status](https://api.netlify.com/api/v1/badges/8309dfde-e5bd-4992-8f78-e3f45c292acf/deploy-status)](https://app.netlify.com/sites/cherab-phix/deploys)

CHERAB for PHiX, which is a small tokamak device in Tokyo Institute of Technology
For more information, see the [documentation pages](https://cherab-phix.netlify.app/).

Quick installation
-------------------
Install it from GitHub repository with pip:

```Shell
python -m pip install git+https://github.com/munechika-koyo/cherab_phix
```

For Developpers
---
If you would like to modificate it, it is much easier to create a conda development environment after cloning repository.
```Shell
conda env create -f environment.yaml
conda activate cherab-phix-dev
python dev.py build
python dev.py install
```


![The plasma discharged image captured by the high-speed camera](docs/_static/images/phix.jpg)

*Caption: The plasma discharged image captured by the high-speed camera at PHiX.*
