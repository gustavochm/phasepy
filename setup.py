from setuptools import setup, Extension, find_packages
import numpy
import os

try:
    from Cython.Build import cythonize
    use_cython = True
except ImportError:
    use_cython = False

ext_modules = []
if use_cython:
    ext_modules = cythonize([
        Extension('phasepy.coloc_cy', ['phasepy/src/coloc_cy.pyx'],
                  include_dirs=[numpy.get_include()]),
        Extension('phasepy.actmodels.actmodels_cy', ['phasepy/src/actmodels_cy.pyx'],
                  include_dirs=[numpy.get_include()]),
        Extension('phasepy.sgt.cijmix_cy', ['phasepy/src/cijmix_cy.pyx'],
                  include_dirs=[numpy.get_include()])
    ], compiler_directives={'language_level': '3'})
else:
    ext_modules = [
        Extension('phasepy.coloc_cy', ['phasepy/src/coloc_cy.c'],
                  include_dirs=[numpy.get_include()]),
        Extension('phasepy.actmodels.actmodels_cy', ['phasepy/src/actmodels_cy.c'],
                  include_dirs=[numpy.get_include()]),
        Extension('phasepy.sgt.cijmix_cy', ['phasepy/src/cijmix_cy.c'],
                  include_dirs=[numpy.get_include()])
    ]

setup(
    packages=find_packages(include=['phasepy', 'phasepy.*']),
    ext_modules=ext_modules,
    include_package_data=True,
    zip_safe=False,
    # This is critical - tells setuptools where to put the compiled extensions
    options={
        'build': {
            'build_lib': 'build/lib',
        },
    }
)