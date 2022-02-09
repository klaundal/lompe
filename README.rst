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

- numpy
- pandas
- scipy
- matplotlib
- `apexpy <https://github.com/aburrell/apexpy>`_
- cartopy
- xarray
- netCDF4 (if you use the DMSP SSUSI preprocessing scripts)
- `pydarn <https://github.com/SuperDARN/pydarn>`_ (if you use the SuperDARN data preprocessing helper scripts)
- madrigalWeb (if you use the DMSP SSIES data preprocessing scripts)
- `astropy <https://github.com/astropy/astropy>`_ (if you use the AMPERE Iridium data preprocessing scripts)
- `cdflib <https://github.com/MAVENSDC/cdflib>`_ (for running lompe paper figure example 05)
You should also have git version >= 2.13


Install
=======
No pip install yet, so you should use git. Clone the repository like this::

    git clone https://github.com/klaundal/lompe

If you have the repository in a place that Python knows (in the PYTHONPATH environent variable), lompe can be imported as a module

We are developing this actively, so it is a good idea to check for newer versions. To get the latest version do::

    git pull

