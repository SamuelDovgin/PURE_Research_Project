import os
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "mutual information estimator"
VERSION = "0.1"
REQUIRES = ['numpy', 'cython']
DESCR = ""
URL = ""
LICENSE = ""
AUTHOR = "Alan Yang"
EMAIL = "asyang2@illinois.edu"


SRC_DIR = "KSG_estimator"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".estimator",
                  [SRC_DIR + "/estimator.pyx"],
                  libraries=[],
                  # libraries=["m"],
                  # extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
