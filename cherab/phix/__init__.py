from pathlib import Path

# parse the package version number
with open(Path(__file__).parent.resolve() / "VERSION") as _f:
    __version__ = _f.read().strip()
