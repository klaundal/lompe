""" functions to calculate the geometry of Sunlight and the meaning of life """

import pandas as pd
import os
import numpy as np
from lompe.utils.time import date_to_doy, is_leapyear
from lompe.dipole.dipole import sph_to_car, car_to_sph, subsol

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


# TODO: this needs work (AØH 17/06/2022)
def terminator(datetime, sza = 90, resolution = 360):
    """ compute terminator trajectory (constant solar zenith angle contour)

        glat, glon = terminator(date, sza = 90, resolution = 360)

        sza is the solar zenith angle contour, default 90 degrees

        return two arrays, geocentric latitude and longitude, which outline the sunlight terminator at given date (datetime)

        does not handle arrays - only one trajectory can be returned at the time

        Method is assuming a spherical geometry:
        - compute the subsolar point, and two approximately duskward and northward normal vectors (these will point at 90 degrees SZA)
        - rotate the northward normal around the duskward normal to get to the correct SZA
        - rotate the resulting vector about the subsolar vector a number of times to trace out the contour.
        - calculate corresponding spherical (geocentric) coordinates

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    sslat, sslon = subsol(datetime)
    #print sslon, sslat
    sslon = sslon[0]*d2r
    sslat = sslat[0]*d2r

    # make cartesian vector
    x = np.cos(sslat) * np.cos(sslon)
    y = np.cos(sslat) * np.sin(sslon) 
    z = np.sin(sslat)
    ss = np.array([x, y, z]).flatten()

    # make a cartesian vector pointing at the pole
    pole = np.array([0, 0, 1])

    # construct a vector pointing roughly towards dusk, and normalize
    t0 = np.cross(ss, pole)
    t0 = t0/np.linalg.norm(t0)

    # make a new vector pointing northward at the 90 degree SZA contour:
    sza90 = np.cross(t0, ss)

    # rotate this about the duskward vector to get specified SZA contour
    rotation_angle = -(sza - 90) * d2r

    sza_vector = sza90 * np.cos(rotation_angle) + np.cross(t0, sza90) * np.sin(rotation_angle) + t0 * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle)) # (rodrigues formula)
    sza_vector = sza_vector.flatten()

    # rotate this about the sun-Earth line to trace out the trajectory:
    angles = np.r_[0 : 2*np.pi: 2*np.pi / resolution][np.newaxis, :]
    r = sza_vector[:, np.newaxis] * np.cos(angles) + np.cross(ss, sza_vector)[:, np.newaxis] * np.sin(angles) + ss[:, np.newaxis] * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle))

    # convert to spherical and return
    tsph = car_to_sph(r, deg = True)

    return 90 - tsph[1], tsph[2]
