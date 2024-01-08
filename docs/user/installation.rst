:orphan:

.. _installation:

============
Installation
============

.. note::

    Currently (1/3/2024), ``cherab-phix`` recommends using python3.9 or 3.10 because there are built
    distributions in the `raysect`_ and `cherab`_ dependencies at PyPI.
    Although the user can take the python3.11+, both `raysect`_ and `cherab`_ must be compiled manually
    from sources.

    The rest of dependencies are listed in ``pyproject.toml`` file in source directory,
    so those who are curious about it should look into it.


Installing using pip
====================
Using ``pip`` command allows us to install ``cherab-phix`` including asset data like device meshes.

.. prompt:: bash

    python -m pip install cherab-phix


Configuring Atomic data
-----------------------
To make use of the PHiX-modeled plasma object, you have to download the atomic data from openadas
repository; run the following commands in a python terminal:

.. prompt:: python >>> auto

    from cherab.openadas.repository import populate
    populate()

If you are curious about it more, please see the ``cherab`` documentation
`here <https://www.cherab.info/installation_and_structure.html#configuring-atomic-data>`_.


Installing for Developper
==========================
If you plan to make any modifications to do any development work on ``cherab-phix``,
and want to be able to edit the source code without having to run the setup script again
to have your changes take effect, you can install ``cherab-phix`` on editable mode.

Manually downloading source
---------------------------
Before install the package, it is required to download the source code from github repository.
The source code can be cloned from the GitHub reporepository with the command:

.. prompt:: bash

    git clone -b development https://github.com/munechika-koyo/cherab_phix

The repository will be cloned inside a new subdirectory named as ``cherab_phix``.

Downloading data asset by Git LFS
---------------------------------
Data assets like device mesh files (``.rsm``, ``.STL``) are stored at the repository
by `Git LFS <https://git-lfs.github.com>`. After installing the Git LFS, the downloading data assets
can be accomplished by the following commands at the source root directory:

.. prompt:: bash

    git lfs install
    git lfs fetch


Editable installation
---------------------
The editable installation is the way to install the package without copying the source codes to
the python site-packages directory by adding the source path to the python import path.
To not violate existing python packages, the editable installation is recommended to be performed
in a virtual environment using ``venv`` or ``conda``.

1. ``venv`` + ``pip``
*********************
The user who is familiar with ``venv`` can use the following commands:

.. prompt:: bash

    python -m venv cherab-phix-dev  # create a virtual environment
    source cherab-phix-dev/bin/activate  # activate the virtual environment
    python -m pip install -vv --no-build-isolation --editable .[dev,doc]  # install the package and dependencies

``--editable`` option allows the user to install the package as the editable mode.

2. ``conda`` + ``pip``
**********************
The user who is familiar with ``conda`` can use the following commands:

.. prompt:: bash

    conda env create -f environment.yaml  # create a virtual environment and install dependencies
    conda activate cherab-phix-dev  # activate the virtual environment
    python -m pip install -vv --no-build-isolation --no-deps --editable .  # install the package

`mamba <https://mamba.readthedocs.io/en/latest/>`_ instead of ``conda`` is the recommended command
to create a virtual environment and install dependencies much faster.


3. ``conda`` + ``dev.py`` CLI
******************************
There is a CLI interface ``dev.py`` to perform all regular development-related tasks
(building, building docs, formatting codes, etc.).
The user can also use this CLI to install the package as the ``setuptools``'s editable mode.
Please try the following commands if there is something wrong with the above commands:

.. prompt:: bash

    conda env create -f environment.yaml  # create a virtual environment and install dependencies
    conda activate cherab-phix-dev  # activate the virtual environment
    python dev.py build  # compile cython codes
    python dev.py install  # install the package as the editable mode


The details of the CLI can be found at :ref:`dev-cli`.
