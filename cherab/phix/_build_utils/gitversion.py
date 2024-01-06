#!/usr/bin/env python3
import sys
from os import environ
from pathlib import Path
from textwrap import dedent

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


def init_version() -> str:
    """Load the version from the pyproject.toml file."""
    init = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
    with init.open("rb") as f:
        pyproject = tomllib.load(f)

    version = pyproject["project"]["version"]

    return version


def git_version(version: str) -> tuple[str, str]:
    """Get the git version if available."""
    import subprocess

    git_hash = ""
    try:
        p = subprocess.Popen(
            ["git", "log", "-1", "--format='%H %aI'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent,
        )
    except FileNotFoundError:
        pass
    else:
        out, _ = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode("utf-8").strip().replace("'", "").split("T")[0].replace("-", "").split()
            )

            # Only attach git tag to development versions
            if "dev" in version:
                version += f"+git{git_date}.{git_hash[:7]}"

    return version, git_hash


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", help="Save version to this file.")
    parser.add_argument(
        "--meson-dist", help="Output path is relative to MESON_DIST_ROOT", action="store_true"
    )
    args = parser.parse_args()

    version, git_hash = git_version(init_version())

    template = dedent(
        f"""
        version = "{version}"
        __version__ = version
        full_version = version

        git_revision = "{git_hash}"
        release = "dev" not in version and "+" not in version
        short_version = version.split("+")[0]
    """
    )

    if args.write:
        outfile = Path(args.write)
        if args.meson_dist:
            outfile = Path(environ.get("MESON_DIST_ROOT", "")) / outfile

        with outfile.open("w") as f:
            print(f"Saving version to {outfile}")
            f.write(template)
    else:
        print(version)
