from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_arma", ["cython_arma.pyx"])],
    include_dirs = [numpy.get_include()]
)
