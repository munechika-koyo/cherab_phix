:orphan:

.. _contribution:

============
Contribution
============

Contributions from any community as well as the fusion community are welcome.
Please feel free to push the PR from the forked repository, post issues, etc.
In the mean time,
interested collaborators should make contact with Koyo Munechika (Core Developer) at Tokyo Tech.
(munechika.koyo@torus.nr.titech.ac.jp)

.. include:: ../../AUTHORS.md


For Developper
--------------
If you would like to develop this package, please fork the GitHub repository at first, and follow
the installation procedure :ref:`here<installation>`.
Additionally, we recommand you should set up the ``pre-commit`` which is the framework to run the
simple code review automatically before a git commit.
After installing development dependencies, ``pre-commit`` is already installed, so simply excute
the following command to complete the configuration:

.. prompt:: bash

    pre-commit install

``pre-commit`` hook is automatically excuted when doing git commits.
If you are curiouse about it more, please see the `pre-commit HP <https://pre-commit.com>`_.


.. _dev-cli:

The ``dev.py`` interface
------------------------
This interface has many options, allowing you to perform all regular development-related tasks
(building docs, formatting codes, etc.).
Here we document a few of the most commonly used options;
run ``python dev.py --help`` or ``--help`` on each of the subcommands for more details.

Use the following command to build the document:

.. prompt:: bash

    python dev.py doc

To lint the cython source codes:

.. prompt:: bash

    python dev.py cython-lint
