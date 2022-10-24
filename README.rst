Overview
========
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
    >>> # now the map is ready, and we can plot plasma flows, currents, magnetic fields, ...
    >>> model.lompeplot()
    >>> # or calculate some quantity, like plasma velocity:
    >>> ve, vn = model.v(mylon, mylat)


Dependencies
============
You should have the following modules installed:

- `apexpy <https://github.com/aburrell/apexpy/>`_
- matplotlib
- numpy
- pandas
- `ppigrf <https://github.com/klaundal/ppigrf/>`_
- scipy
- xarray
- `astropy <https://github.com/astropy/astropy/>`_ (if you use the AMPERE Iridium data preprocessing scripts)
- `cdflib <https://github.com/MAVENSDC/cdflib/>`_ (for running lompe paper figures example 05)
- `madrigalWeb <https://pypi.org/project/madrigalWeb/>`_ (if you use the DMSP SSIES data preprocessing scripts)
- `netCDF4 <https://github.com/Unidata/netcdf4-python/>`_ (if you use the DMSP SSUSI data preprocessing scripts)
- `pyAMPS <https://github.com/klaundal/pyAMPS/>`_ (for running code paper figures example 08)
- `pydarn <https://github.com/SuperDARN/pydarn/>`_ (if you use the SuperDARN data preprocessing scripts)

You should also have git version >= 2.13


Install
=======
No pip install yet, so you should use git. But you'll need to install a module called ppigrf, which calculates IGRF values. That can be done like this::

    pip install ppigrf

Clone the repository like this::

    git clone https://github.com/klaundal/lompe

Then, to get the code for the submodules, run::

    git submodule init
    git submodule update

If you have the repository in a place that Python knows (in the PYTHONPATH environent variable), lompe can be imported as a module

We are developing this actively, so it is a good idea to check for newer versions. To get the latest version do::

    git pull


Lompe papers
============
Main Lompe paper that describes the technique: `Local Mapping of Polar Ionospheric Electrodynamics <https://doi.org/10.1029/2022JA030356>`_

Funding
=======
The Lompe development is funded by the `Trond Mohn Foundation <https://birkeland.uib.no/trond-mohn-stiftelse-grant/>`_, and by the Research Council of Norway (300844/F50)
