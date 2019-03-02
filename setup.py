from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [Extension('phasepy.coloc_cy', ['phasepy/src/coloc_cy.pyx']),
                   Extension('phasepy.actmodels.actmodels_cy', ['phasepy/src/actmodels_cy.pyx']),
                    Extension('phasepy.sgt.cijmix_cy', ['phasepy/src/cijmix_cy.pyx'])]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules +=  [Extension('phasepy.coloc_cy', ['phasepy/src/coloc_cy.c']),
                   Extension('phasepy.actmodels.actmodels_cy', ['phasepy/src/actmodels_cy.c']),
                    Extension('phasepy.sgt.cijmix_cy', ['phasepy/src/cijmix_cy.c'])]

long_description = "Phasepy is a open-source scientific python package for fluid phase equilibria computation.
This package facilitate the calculation of liquid-vapour equilibrium, liquid-liquid equilibrium
and liquid-liquid-vapour equilibrium. Equilibrium calculations can be perfomed with cubic equations
of state with clasic or advances mixing rules or with a discontinuous approach using a virial equations
of state for the vapour phase and a activity coefficient model for the liquid phase.

Besides computations, with this package is also possible to fit phase equilibria data, functions to fit quadratic
mix rule, NRTL, Wilson and Redlich Kister parameters, are included.

Phasety relys on numpy, scipy and cython extension modules, when necessary.
"
setup(
  name = 'phasepy',
  license='MIT',
  version = '0.0.4',
  description = 'Multiphase multicomponent Equilibria',
  author = 'Gustavo Chaparro Maldonado, Andrés Mejía Matallana',
  author_email = 'gustavochaparro@udec.cl',
  url = 'https://github.com/gustavochm/phasepy',
  download_url = 'https://github.com/gustavochm/phasepy.git',
  long_description = long_description,
  packages = ['phasepy', 'phasepy.cubic', 'phasepy.equilibrium','phasepy.fit', 'phasepy.sgt', 'phasepy.actmodels'],
  cmdclass = cmdclass,
  ext_modules = ext_modules,
  install_requires=['numpy','scipy', 'pandas'],
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  keywords = ['Phase Equilibrium', 'Cubic EOS', 'QMR' , 'MHV', 'NRTL', 'Wilson', 'UNIFAC', 'Flash', 'LVE', 'LLE' , 'LLVE'],
  package_data={'phasepy': ['database/*']},
  zip_safe=False
)