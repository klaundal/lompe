"""
Functions for conversion between spherical and Cartesian coordinates



MIT License

Copyright (c) 2017 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

d2r = np.pi/180
r2d = 180/np.pi



def sph_to_car(sph, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    deg : bool, optional
        set to True if input is given in degrees. False if radians

    Returns
    -------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates
    """

    r, theta, phi = sph

    conv = 1 if deg == False else d2r

    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))

def car_to_sph(car, deg = True):
    """ Convert from spherical to cartesian coordinates

    Parameters
    ----------
    car : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        x, y, z, in ECEF coordinates

    Returns
    -------
    sph : 3 x N array
        3 x N array, where the rows are, from top to bottom:
        radius, colatitude, and longitude
    deg : bool, optional
        set to True if output is wanted in degrees. False if radians
    """

    x, y, z = car

    conv = 1 if deg == False else d2r

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)*conv
    phi = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    return np.vstack((r, theta, phi))


def enu_to_ecef(v, lon, lat, reverse = False):
    """ convert vector(s) v from ENU to ECEF (or opposite)

    Parameters
    ----------
    v: array
        N x 3 array of east, north, up components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)
    reverse: bool (optional)
        perform the reverse operation (ecef -> enu). Default False

    Returns
    -------
    v_ecef: array
        N x 3 array of x, y, z components


    Author: Kalle, March 2020
    """

    # construct unit vectors in east, north, up directions:
    ph = lon * d2r
    th = (90 - lat) * d2r

    e = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
    n = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
    u = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

    # rotation matrices (enu in columns if reverse, in rows otherwise):
    R_EN_2_ECEF = np.stack((e, n, u), axis = 1 if reverse else 2) # (N, 3, 3)

    # perform the rotations:
    return np.einsum('nij, nj -> ni', R_EN_2_ECEF, v)


def ecef_to_enu(v, lon, lat):
    """ convert vector(s) v from ECEF to ENU

    Parameters
    ----------
    v: array
        N x 3 array of x, y, z components
    lat: array
        N array of latitudes (degrees)
    lon: array
        N array of longitudes (degrees)

    Returns
    -------
    v_ecef: array
        N x 3 array of east, north, up components

    See enu_to_ecef for implementation details
    """
    return enu_to_ecef(v, lon, lat, reverse = True)

