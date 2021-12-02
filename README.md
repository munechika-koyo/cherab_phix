CHERAB-PHiX
============

CHERAB for PHiX, which is a small tokamak device in Tokyo Institute of Technology
For more information, see the [documentation pages](https://cherab-phix.netlify.app/).

Quick installation
-------------------
Synchronize cherab_phix:

```Shell
git clone https://github.com/munechika-koyo/cherab_phix.git
```

And compile it:
```Shell
python setup.py install build_ext --inplace
```

If you would like to develop this package, it is preferred to use "develop" mode:

```Shell
python setup.py develop build_ext --inplace
```