:orphan:

.. _installation:

============
Installation
============

.. note::

    Currently (12/20/2022), ``cherab-phix`` recommends using python3.8 or 3.9 because there are built
    distributions in the `raysect`_ and `cherab`_ dependencies at PyPI.
    Although the user can take the python3.10+, both `raysect`_ and `cherab`_ must be compiled manually
    from sources.

    The rest of dependencies are listed in ``pyproject.toml`` file in source directory,
    so those who are curious about it should look into it.


Installing using pip
====================
Using ``pip`` command allows us to install ``cherab-phix`` including asset data like a device meshes.

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
The source codes can be cloned from the GitHub reporepository with the command:
Before install the package, it is required to download the source code from github repository.

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


Building and Installing
-----------------------
For pip users, run the following command:

.. prompt:: bash

    python -m pip install -e .[dev,doc]

``-e`` or ``--editable`` option allows the user to install the package as the editable mode.

For conda users, creating a virtual environment is the most useful process:

.. prompt:: bash

    conda env create -f environment.yaml  # `mamba` works too for this command
    conda activate cherab-phix-dev

And the same pip command above enables the user to install the package.

Alternatively, the ``dev.py`` CLI is available:

.. prompt:: bash

    python dev.py build
    python dev.py install

These commands enable the user to compile cython codes and install it as the editable mode.
This interface has some options, allowing you to perform all regular development-related tasks
(building, building docs, formatting codes, etc.).
Here we document a few of the most commonly used options; run ``python dev.py --help`` or ``--help``
on each of the subcommands for more details.
