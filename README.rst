Overview
========

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/klaundal/lompe/main

LOcal Mapping of Polar ionospheric Electrodynamics (Lompe)

Lompe is a tool for estimating regional maps of ionospheric electrodynamics using measurements of plasma convection and magnetic field disturbances in space and on ground. 

We recommend to use the examples to learn how to use Lompe, but the general workflow is like this:

.. code-block:: python

    >>> # prepare datasets (as many as you have - see lompe.Data doc string for how to format)
    >>> my_data1 = lompe.Data(*data1)
    >>> my_data2 = lompe.Data(*data2)
    >>> # set up grid (the parameters depend on your region, target resoultion etc):
    >>> grid = lompe.cs.CSgrid(lompe.cs.CSprojection(*projectionparams), *gridparams)
    >>> # initialize model with grid and functions to calculate Hall and Pedersen conductance
    >>> # The Hall and Pedersen functions should take (lon, lat) as parameters
    >>> model = lompe.Emodel(grid, (Hall_function, Pedersen_function))
    >>> # add data:
    >>> model.add_data(my_data1, my_data2)
    >>> # run inversion
    >>> model.run_inversion()
    >>> # now the model vector is ready, and we can plot plasma flows, currents, magnetic fields, ...
    >>> model.lompeplot()
    >>> # or calculate some quantity, like plasma velocity:
    >>> ve, vn = model.v(mylon, mylat)

Install
=======

(NB: In the below, if you do not have mamba, replace `mamba` with `conda`)

Option 0: using pip directly
----------------------------

The package is pip-installable from GitHub directly with::

    pip install "lompe[deps-from-github,extras] @ git+https://github.com/klaundal/lompe.git@main"

You can omit some of the optional packages by removing ``,extras``.

This could also be done within a minimal conda environment created with, e.g. ``mamba create -n lompe python=3.10 fortran-compiler``

Option 1: without development install of dipole, polplot, secsy
---------------------------------------------------------------

Get the code, create a suitable conda environment, then use pip to install the package in editable (development) mode::

    git clone https://github.com/klaundal/lompe
    mamba env create -f lompe/binder/environment.yml -n lompe
    mamba activate lompe
    pip install --editable ./lompe[extras,deps-from-github]

Editable mode (``-e`` or ``--editable``) means that the install is directly linked to the location where you cloned the repository, so you can edit the code.

Note that in this case, the ``deps-from-github`` option means that the ``dipole, polplot, secsy`` packages are installed directly from their source on GitHub.

Option 2: including development install of dipole, polplot, secsy
-----------------------------------------------------------------

Get all the repositories, create a suitable conda environment, then use pip to install all of them in editable (development) mode::

    git clone https://github.com/klaundal/dipole
    git clone https://github.com/klaundal/polplot
    git clone https://github.com/klaundal/secsy
    git clone https://github.com/klaundal/lompe
    mamba env create -f lompe/binder/environment.yml -n lompe
    mamba activate lompe
    pip install -e ./dipole -e ./secsy -e ./polplot -e ./lompe[local,extras]

Note that in this case, all four are installed in editable mode. And the ``local`` option instructs the lompe install to use those local versions of the package.

Hint: you can use ``pip list | grep -E 'dipole|polplot|secsy|lompe'`` to identify which versions you are using.

Hint: you can use ``pytest ./lompe/tests`` to check it installed correctly.

Dependencies
============
You should have the following modules installed:

- `apexpy <https://github.com/aburrell/apexpy/>`_
- matplotlib
- numpy
- pandas
- `ppigrf <https://github.com/klaundal/ppigrf/>`_ (install with pip install ppigrf)
- scipy
- xarray
- `astropy <https://github.com/astropy/astropy/>`_ (if you use the AMPERE Iridium data preprocessing scripts)
- `cdflib <https://github.com/MAVENSDC/cdflib/>`_ (for running lompe paper figures example 05)
- `madrigalWeb <https://pypi.org/project/madrigalWeb/>`_ (if you use the DMSP SSIES data preprocessing scripts)
- `netCDF4 <https://github.com/Unidata/netcdf4-python/>`_ (if you use the DMSP SSUSI data preprocessing scripts)
- `pyAMPS <https://github.com/klaundal/pyAMPS/>`_ (for running code paper figures example 08)
- `pydarn <https://github.com/SuperDARN/pydarn/>`_ (if you use the SuperDARN data preprocessing scripts)

You should also have git version >= 2.13

Lompe papers
============
- Main Lompe paper that describes the technique: `Local Mapping of Polar Ionospheric Electrodynamics <https://doi.org/10.1029/2022JA030356>`_
- Paper about the Lompe code: `The Lompe code: A Python toolbox for ionospheric data analysis <https://doi.org/10.3389/fspas.2022.1025823>`_

Funding
=======
The Lompe development is funded by the `Trond Mohn Foundation <https://birkeland.uib.no/trond-mohn-stiftelse-grant/>`_, and by the Research Council of Norway (300844/F50)
