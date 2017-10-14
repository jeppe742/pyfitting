#from distutils.core import setup
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'Pyfitting',

    version = 0.9,

    long_description = "Linear and non-linear fitting in python, made easy",
    url = "https://github.com/jeppe742/pyfit",
    author = "Jeppe Johan Waarkjær Olsen",
    install_requires=['autodiff'],
    packages = find_packages(),

)