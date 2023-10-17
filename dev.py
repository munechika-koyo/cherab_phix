"""Developper CLI: building sources in place, document, etc."""
import os
import subprocess
import sys
from pathlib import Path

import rich_click as click

try:
    import tomllib
except ImportError:
    import tomli as tomllib


BASE_DIR = Path(__file__).parent.absolute()
SRC_PATH = BASE_DIR / "cherab"
DOC_SRC = BASE_DIR / "docs"
ENVS = dict(os.environ)
N_CPUs = os.cpu_count()


@click.group()
def cli():
    """Developper CLI: building sources in place, document, etc."""
    pass


############


@cli.command()
@click.option(
    "-j",
    "--parallel",
    default=N_CPUs,
    show_default=True,
    help="Number of parallel jobs for building.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Forces the recompilation of the Cython modules.",
)
def build(parallel: int, force: bool):
    """Build package with compiling cython codes.

    \b
    ```python
    Examples:
    $ python dev.py build
    ```
    """
    # === build project =============================================
    cmd = ["python", "setup.py", "build_ext", f"-j{parallel}", "--inplace"]
    if force:
        cmd += ["--force"]

    click.echo(" ".join([str(p) for p in cmd]))

    os.chdir(BASE_DIR)
    ret = subprocess.call(cmd)

    if ret == 0:
        print("Build OK")
    else:
        print("Build failed!")
        sys.exit(1)


@cli.command()
def install():
    """Install package as the editalbe mode.

    This command enables us to install the packages
    as an editable mode with the setuptools functionality.

    \b
    ```python
    Examples:
    $ python dev.py install
    ```
    """
    # install the package
    cmd = [sys.executable, "setup.py", "develop"]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("Successfully installed.")
    else:
        print("install failed!")
        sys.exit(1)


@cli.command()
def install_deps():
    """Install build dependencies using pip.

    Only pip install cannot compile cython files appropriately, so we excute this command before
    installing this package.
    """
    # Load requires from pyproject.toml
    pyproject = BASE_DIR / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError("pyproject.toml must be placed at the root directory.")

    with open(pyproject, "rb") as file:
        conf = tomllib.load(file)
    requires = conf["build-system"].get("requires")
    subprocess.run([sys.executable, "-m", "pip", "install"] + requires)


############


@cli.command()
@click.argument("targets", default="html")
@click.option(
    "-j",
    "--parallel",
    default=N_CPUs,
    show_default=True,
    help="Number of parallel jobs for building.",
)
def doc(parallel: int, targets: str):
    """:wrench: Build documentation
    TARGETS: Sphinx build targets [default: 'html']
    """
    # move to docs/ and run command
    os.chdir("docs")

    builddir = DOC_SRC / "build"
    SPHINXBUILD = "sphinx-build"
    if targets == "html":
        cmd = [SPHINXBUILD, "-b", targets, f"-j{parallel}", str(DOC_SRC), str(builddir / "html")]

    elif targets == "clean":
        cmd = ["rm", "-rf", str(builddir), "&&", "rm", "-rf", str(DOC_SRC / "_api")]

    elif targets == "help":
        cmd = [SPHINXBUILD, "-M", targets, str(DOC_SRC), str(builddir)]

    else:
        cmd = [SPHINXBUILD, "-M", targets, f"-j{parallel}", str(DOC_SRC), str(builddir)]

    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("sphinx-build successfully done.")
    else:
        print("Sphinx-build has errors.")
        sys.exit(1)


############


@cli.command()
def format():
    """:art: Run ruff linting & formatting The default options are defined in pyproject.toml."""
    cmd = ["ruff", "check", "--fix", str(SRC_PATH)]
    click.echo(" ".join([str(p) for p in cmd]))
    ret = subprocess.call(cmd)

    if ret == 0:
        print("ruff formated")
    else:
        print("ruff formatting errors!")
        sys.exit(1)


@cli.command()
def cython_lint():
    """:art: Cython linter.

    Checking all .pyx files in the source directory. The default options are defined at the cython-
    lint table in pyproject.toml
    """

    # list of .pyx files
    pyx_files = [str(pyx_path) for pyx_path in SRC_PATH.glob("**/*.pyx")]

    cmd = ["cython-lint"] + pyx_files
    ret = subprocess.call(cmd)

    if ret == 0:
        print("cython-lint OK")
    else:
        print("cython-lint errors")
        sys.exit(1)


#######
def config(tool: str):
    """Load configure data from pyproject.toml for tool table."""
    pyproject = BASE_DIR / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError("pyproject.toml must be placed at the root directory.")

    with open(pyproject, "rb") as file:
        conf = tomllib.load(file)

    if not conf["tool"].get(tool):
        raise ValueError(f"{tool} config data does not exist.")

    return conf["tool"].get(tool)


if __name__ == "__main__":
    cli()
