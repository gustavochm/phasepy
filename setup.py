from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [Extension('phasepy.coloc_cy',
                              ['phasepy/src/coloc_cy.pyx']),
                    Extension('phasepy.actmodels.actmodels_cy',
                              ['phasepy/src/actmodels_cy.pyx']),
                    Extension('phasepy.sgt.cijmix_cy',
                              ['phasepy/src/cijmix_cy.pyx'])]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [Extension('phasepy.coloc_cy', ['phasepy/src/coloc_cy.c']),
                    Extension('phasepy.actmodels.actmodels_cy',
                              ['phasepy/src/actmodels_cy.c']),
                    Extension('phasepy.sgt.cijmix_cy',
                              ['phasepy/src/cijmix_cy.c'])]


setup(
  name='phasepy',
  license='MIT',
  version='0.0.50',
  description='Multiphase multicomponent Equilibria',
  author='Gustavo Chaparro Maldonado, Andres Mejia Matallana',
  author_email='gustavochaparro@udec.cl',
  url='https://github.com/gustavochm/phasepy',
  download_url='https://github.com/gustavochm/phasepy.git',
  long_description=open('long_description.rst').read(),
  packages=['phasepy', 'phasepy.cubic', 'phasepy.equilibrium', 'phasepy.fit',
            'phasepy.sgt', 'phasepy.actmodels'],
  cmdclass=cmdclass,
  ext_modules=ext_modules,
  install_requires=['numpy', 'scipy', 'pandas'],
  platforms=["Windows", "Linux", "Mac OS", "Unix"],
  keywords=['Phase Equilibrium', 'Cubic EOS', 'QMR', 'MHV', 'WS', 'NRTL',
            'Wilson', 'UNIFAC', 'UNIQUAC', 'Flash', 'VLE', 'LLE', 'VLLE',
            'SGT'],
  package_data={'phasepy': ['database/*']},
  zip_safe=False
)
