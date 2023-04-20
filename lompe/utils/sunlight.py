""" functions to calculate the geometry of Sunlight and the meaning of life """

import pandas as pd
import os
import numpy as np
from lompe.utils.time import date_to_doy, is_leapyear
from dipole import sph_to_car, car_to_sph, subsol

d2r = np.pi / 180
r2d = 180 / np.pi


def sza(glat, glon, datetimes, degrees = True):
    """ calculate solar zenith angle at given latitude (not colat), longitude and datetimes

        handles arrays, but does not preserve shape - a flat output is returned

        the following combinations are possible:

        1) glat, glon arrays of same size and datetimes scalar
           output will be array with same size as glat, glon
        2) glat, glon, datetimes arrays of same size
           output will be array with same size as glat, glon, datetimes
        3) glat, glon scalar and datetimes array
           output will be array with same size as datetimes


        Spherical geometry is assumed

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    glat = np.array(glat, ndmin = 1).flatten() # turn into array and flatten
    glon = np.array(glon, ndmin = 1).flatten() # turn into array and flatten

    if glat.size != glon.size:
        raise ValueError('sza: glat and glon arrays but not of same size')

    if hasattr(datetimes, '__iter__'):
        if len(datetimes) != len(glat) and len(glat) != 1:
            raise ValueError('sza: inconsistent input size')

    if degrees:
        conv = r2d
    else:
        conv = 1.

    # compute subsolar point
    sslat, sslon = subsol(datetimes)

    # compute and return the angle
    ssr = sph_to_car(np.vstack((np.ones_like(sslat), 90. - sslat, sslon)), deg = True)
    gcr = sph_to_car(np.vstack((np.ones_like(glat ), 90. - glat , glon )), deg = True)

    # the angle is arccos of the dot product of these two vectors
    return np.arccos(np.sum(ssr*gcr, axis = 0))*conv