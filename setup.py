#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name = 'Pyfitting',

    version = "0.9.1",

    long_description = "Linear and non-linear fitting in python, made easy",
    url = "https://github.com/jeppe742/pyfitting",
    author = "Jeppe Johan Waarkjaer Olsen",
    install_requires=['autodiff', 'numpy', 'matplotlib'],
    packages = find_packages(),

)