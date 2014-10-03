#!/usr/bin/env python
import sys
import os

if len(sys.argv) == 1:
    sys.argv.append('install')

# Use our own nose-based test harness
if sys.argv[1] == 'test':
    from subprocess import call
    sys.exit(call([sys.executable, 'test.py']+sys.argv[2:]))

#sys.dont_write_bytecode = True

#from distutils.core import Extension
from setuptools import setup, find_packages, Extension
#import fix_setuptools_chmod

import refl1d


packages = find_packages()

    
# reflmodule extension
def reflmodule_config():
    import distutils.ccompiler
    compiler = distutils.ccompiler.get_default_compiler()
    for arg in sys.argv:
        if arg.startswith('--compiler='):
            _,compiler=arg.split('=')
    if compiler == 'msvc':
        eca=['/EHsc']
    else:
        eca=['']

    S = ("reflmodule.cc","methods.cc",
         "reflectivity.cc", "magnetic.cc",
         "contract_profile.cc",
         "convolve.c", "convolve_sampled.c",
    )

    Sdeps = ("erf.c","methods.h","rebin.h","rebin2D.h","reflcalc.h")
    sources = [os.path.join('refl1d','lib',f) for f in S]
    depends = [os.path.join('refl1d','lib',f) for f in Sdeps]
    module = Extension('refl1d.reflmodule',
                       sources=sources, depends=depends, 
                       extra_compile_args=eca )
    return module

from numpy import get_include
# SCF extension
def SCFmodule_config():
    return Extension('refl1d.calc_g_zs_cex', 
                     sources=[os.path.join('refl1d','lib','calc_g_zs_cex.c')],
                     include_dirs=[get_include()])
                     
#TODO: write a proper dependency checker for packages which cannot be
# installed by easy_install
#dependency_check('numpy>=1.0', 'scipy>=0.6', 'matplotlib>=1.0', 'wx>=2.8.9')

#sys.dont_write_bytecode = False
dist = setup(
        name = 'refl1d',
        version = refl1d.__version__,
        author='Paul Kienzle',
        author_email='pkienzle@nist.gov',
        url='http://www.reflectometry.org/danse/model1d.html',
        description='1-D reflectometry modelling',
        long_description=open('README.txt').read(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: Public Domain',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        packages = packages,
        #package_data = gui_resources.package_data(),
        scripts = ['bin/refl1d_cli.py','bin/refl1d_gui.py'],
        ext_modules = [reflmodule_config(),SCFmodule_config()],
        install_requires = ['bumps>=0.7.5','periodictable'],
        )

# End of file
