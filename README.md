# CHERAB-PHiX

![PyPI](https://img.shields.io/pypi/v/cherab-phix?logo=PyPI)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cherab-phix?logo=Python)

[![DOI](https://zenodo.org/badge/239309930.svg)](https://zenodo.org/badge/latestdoi/239309930)
![GitHub](https://img.shields.io/github/license/munechika-koyo/cherab_phix)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_phix/master.svg)](https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_phix/master)
[![Documentation Status](https://readthedocs.org/projects/cherab-phix/badge/?version=stable)](https://cherab-phix.readthedocs.io/en/stable/?badge=stable)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Docstring formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Docstring style: numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


CHERAB for PHiX, which is a small tokamak device in Tokyo Institute of Technology
For more information, see the [documentation pages](https://cherab-phix.readthedocs.io/).

Quick installation
-------------------
Install it from GitHub repository with pip:

```Shell
python -m pip install cherab-phix
```

For Developpers
---
If you would like to modificate codes, it is much easier to create a conda development environment after cloning repository.
```Shell
conda env create -f environment.yaml
conda activate cherab-phix-dev
python dev.py build
python dev.py install
```
And please follows the development [specified procedure](https://cherab-phix.readthedocs.io/en/development/user/contribution.html).

![The plasma discharged image captured by the high-speed camera](docs/_static/images/phix.jpg)

*Caption: The plasma discharged image captured by the high-speed camera at PHiX.*
