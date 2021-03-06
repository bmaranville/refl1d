#!/bin/sh

# should be testing python 2.6, 2.7, 3.3, 3.4
# To set up python 3 environment:
# sudo apt-get install python3 python3-numpy python3-scipy python3-matplotlib python3-pip python3-nose python3-sphinx
PYTHON=${PYTHON:-python}
PYTHON3=${PYTHON3:-python3}
export PYTHON PYTHON3
set -x

# Pull the latest bumps/periodictable from github, rather than pypi.
# A true bleeding edge build would also install the latest versions of
# the various dependent packages (numpy, scipy, numdifftools, pyparsing,
# etc.), but that is a project for a different day.
#$PYTHON -m pip install --no-deps -t external https://github.com/bumps/bumps/tarball/master
#$PYTHON -m pip install --no-deps -t external https://github.com/pkienzle/periodictable/tarball/master
rm -rf external
mkdir external
(cd external && git clone https://github.com/bumps/bumps.git)
(cd external && git clone https://github.com/pkienzle/periodictable.git)

PYTHONPATH=$WORKSPACE/external/bumps:$WORKSPACE/external/periodictable:$PYTHONPATH
export PYTHONPATH

echo $PYTHONPATH
# build and test
$PYTHON setup.py build  || exit 1
$PYTHON test.py || exit 1
$PYTHON check_examples.py --chisq || exit 1

$PYTHON3 setup.py build  || exit 1
$PYTHON3 test.py || exit 1
$PYTHON3 check_examples.py --chisq || exit 1

# check that the docs build
# only need to do this on one version of python.
(cd doc && make html pdf) || exit 1
cp doc/_build/latex/Refl1D.pdf . || exit 1
