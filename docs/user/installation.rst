:orphan:

.. _installation:

============
Installation
============


Installing using pip
====================
Using ``pip`` command allows us to install cherab-phix including dependencies.
For now, it is only available from `Cherab-phix's GitHub repository`_.

.. prompt:: bash

    python -m pip install git+https://github.com/munechika-koyo/cherab_phix



Installing for Developper
==========================
If you plan to make any modifications to do any development work on CHERAB-PHiX,
and want to be able to edit the source code without having to run the setup script again
to have your changes take effect, you can install CHERAB-PHiX on editable mode.

manually download source
------------------------
The source codes can be cloned from the GitHub reporepository with the command:
Before install the package, it is required to download the source code from github repository.

.. prompt:: bash

    git clone https://github.com/munechika-koyo/cherab_phix

The repository will be cloned inside a new subdirectory named as ``cherab_phix``.

Building and Installing
-----------------------
For pip users, run the following command:

.. prompt:: bash

    python -m pip install -e .[dev,doc]

``-e`` or ``--editable`` option allows the user to install the package as the editable mode.

For conda users, creating a conda development environment is the most useful process:

.. prompt:: bash

    conda env create -f environment.yaml  # `mamba` works too for this command
    conda activate cherab-phix-dev

And the same pip command above enables the user to install the package.

Alternatively, the ``dev.py`` CLI is available:

.. prompt:: bash

    python dev.py build
    python dev.py install

These commands enable the user to compile cython codes and install it as editable mode.
This interface has some options, allowing you to perform all regular development-related tasks
(building, building docs, formatting codes, etc.).
Here we document a few of the most commonly used options; run ``python dev.py --help`` or ``--help``
on each of the subcommands for more details.
