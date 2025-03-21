.. _contributing:

********************
Contributing Changes
********************

.. contents:: :local:


The refl1d package is a community project, and we welcome contributions from anyone.  
The package is developed collaboratively on `Github <https://github.com>`_ - if you don't have an account yet, you can sign up for free. For direct write access to the repository, it is required that your accout have `two-factor authentication enabled <https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa>`_.
You may also want to configure your account to use `SSH keys <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_ for authentication.

The best way to contribute to the reflectometry package is to work
from a copy of the source tree in the revision control system.

The refl1d project is hosted on github at:

    https://github.com/reflectometry/refl1d

You will need the git source control software for your computer.  This can
be downloaded from the `git page <http://www.git-scm.com/>`_, or you can use
an integrated development environment (IDE) such as Eclipse and PyCharm, which
may have git built in.


Getting the Code
================

To get the code, you will need to clone the repository.  If you are planning
on making only a few small changes, you can clone the repository directly,
make the changes, document and test, then send a patch (see `Simple patches`_ below).

If you are planning on making larger changes, you should fork the repository
on github, make the changes in your fork, then issue a pull request to the
main repository (see `Larger changes`_ below).

.. note::

    If you are working on a fork, the clone line is slightly different::
            
        git clone https://github.com/YourGithubAccount/refl1d

    
    You will also need to keep your fork up to date
    with the main repository.  You can do this by adding the main repository
    as a remote, fetching the changes, then merging them into your fork.

    .. code-block:: bash

        # Add the main repository as a remote
        git remote add refl1d

        # Fetch the changes from the main repository
        git fetch refl1d

        # Merge the changes into your fork
        git merge refl1d/master

        # Push the changes to your fork
        git push


Once you have the code, you will need to install Refl1D, including its dependencies.
You can do this by following the instructions in the `Installation guide <install.html>`_.

Making Changes
==============

Pre-commit hooks
----------------

Once you have installed the code with ``pip install -e .[dev]``, you can make changes to the code. Note that refl1d uses `pre-commit <https://pre-commit.com/>`_ to run automated checks and linting/formatting on the code before it is committed.

First, activate the Python environment in which you installed refl1d. Then, install the pre-commit hooks by running::

    pre-commit install

This will install the pre-commit hooks in your git repository. The pre-commit hooks will run every time you commit changes to the repository. If the checks fail, the commit will be aborted. 

You can run the checks manually by running::

    pre-commit run

To see what actions are being run, inspect the `.pre-commit-config.yaml` file in the root of the repository.

Simple patches
--------------

If you want to make one or two tiny changes, it is easiest to clone the
repository, make the changes, then send a patch.  This is the simplest way
to contribute to the project.

To run the package from the source tree use the following::

    cd refl1d
    python run.py

This will first build the package into the build directory then run it.
Any changes you make in the source directory will automatically be used in
the new version.

As you make changes to the package, you can see what you have done using git::

    git status
    git diff

Please update the documentation and add tests for your changes.  We use
doctests on all of our examples so that we know our documentation is correct.
More thorough tests are found in test directory. You can run these tests via pytest, 
or via the convenience Makefile target::

    pytest 
    # or
    make test

When all the tests run, create a patch and send it to paul.kienzle@nist.gov::

    git diff > patch

Larger changes
--------------

For a larger set of changes, you should fork refl1d on github, and issue pull
requests for each part.

After you have tested your changes, you will need to push them to your github
fork::

    git commit -a -m "short sentence describing what the change is for"
    git push

Good commit messages are a bit of an art.  Ideally you should be able to
read through the commit messages and create a "what's new" summary without
looking at the actual code.

Make sure your fork is up to date before issuing a pull request.  You can
track updates to the original refl1d package using::

    git remote add refl1d https://github.com/reflectometry/refl1d
    git fetch refl1d
    git merge refl1d/master
    git push

When making changes, you need to take care that they work on different
versions of python. Using conda makes it convenient to maintain multiple independent
environments. You can create a new environment for testing with, for example::

    conda create -n py312 python=3.12
    conda activate py312
    pip install -e .[dev]

Even better is to test against all current python versions::

    pythonX.Y tests.py
    pythonX.Y run.py

When all the tests run, issue a pull request from your github account.

Building Documentation
======================

Building the package documentation requires a working Sphinx installation,
and latex to build the pdf. As of this writing we are using sphinx 8.0.2.

The command line to build the docs is as follows::

    (cd doc && make clean html pdf)

You can see the result by pointing your browser to::

    doc/_build/html/index.html
    doc/_build/latex/Refl1d.pdf

On Windows, you may first need to install `mingw32-make` via conda, 
or simply run sphinx directly from python::

    cd doc
    python -m sphinx.__init__ -b html -d _build/doctrees . _build/html

ReStructured text format does not have a nice syntax for superscripts and
subscripts.  Units such as |g/cm^3| are entered using macros such as
\|g/cm^3| to hide the details.  The complete list of macros is available in

        doc/sphinx/rst_prolog

In addition to macros for units, we also define cdot, angstrom and degrees
unicode characters here.  The corresponding latex symbols are defined in
doc/sphinx/conf.py.

There is a bug in older sphinx versions (e.g., 1.0.7) in which latex tables
cannot be created.  You can fix this by changing::

    self.body.append(self.table.colspec)

to::

    self.body.append(self.table.colspec.lower())

in site-packages/sphinx/writers/latex.py.

Windows Installer
=================

You can build the standalone executable using the powershell script::

    extra\\build_win_installer.ps1

This creates the distribution archive in the dist directory, including
python, the application, the supporting libraries and everything else needed
to run the application.

The installer build script is run automatically on github in response
to a checkin on the master branch via GitHub Actions.

OS/X Installer
==============

A Python script is available to build the OS/X installer::

    extra/build_dmg.py

This script builds a `.dmg` based on the contents of the `dist/<product version>.app` directory.
It can be called with the name and version of the product as arguments, e.g.::

    python extra/build_dmg.py Refl1D 0.8.17

This script is also run automatically on github in response
to a checkin on the master branch via GitHub Actions.

Creating a new release
----------------------

A developer with maintainer status can tag a new release and publish a package to the `Python
Package Index (PyPI) <https://pypi.org/project/refl1d/>`_. Refl1d uses
`versioningit <https://versioningit.readthedocs.io/>`_ to generate the version number
from the latest tag in the git repository.

1. Update the local copy of the master branch::

    # update information from all remotes
    git fetch -p -P -t --all
    # update local copy of master
    git checkout master
    git rebase origin/master
    # check the current version number
    versioningit
    > 0.8.17.dev805

2. Add release notes and commit to master.

3. Create the new tag and push it to the remote. Pushing a tag starts the GitHub workflow job to
publish to PyPI (defined in `.github/workflows/publish.yml
<https://github.com/reflectometry/refl1d/blob/master/.github/workflows/publish.yml>`_)::

    git tag v1.0.0
    versioningit
    > 1.0.0
    git push origin --tags master
