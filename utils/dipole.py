"""
Functions for calculating dipole axis and dipole poles at given epoch(s)
using IGRF Gauss coefficients. 

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
import pandas as pd
from lompe.ppigrf.ppigrf import read_shc
from .coords import car_to_sph, sph_to_car, enu_to_ecef, ecef_to_enu

RE = 6371.2 # reference radius in km

d2r = np.pi/180
r2d = 180/np.pi

# load the IGRF dipole coefficients:
g, h = read_shc()
assert np.all((g.index.day == 1) & (g.index.month == 1)) # IGRF coefficients should be defined for first day of year

igrf_dipole = pd.DataFrame({'g10':g[(1, 0)].values, 'g11':g[(1, 1)].values, 'h11':h[(1, 1)].values}, index = np.float32(g.index.year))
igrf_dipole['B0'] = np.sqrt(igrf_dipole['g10']**2 + igrf_dipole['g11']**2 + igrf_dipole['h11']**2)



def dipole_field(mlat, r, epoch = 2020):
    """ calculate components of the dipole field in dipole coordinates 

    The dipole moment will be calculated from IGRF coefficients at given epoch

    Parameters
    ----------
    mlat : array
        magnetic latitude (latitude in a system with dipole pole at pole)
    r : array
        radius in km
    epoch : float, optional
        The dipole moment will be calculated from IGRF coefficients at given epoch
        default epoch is 2020
    
    Returns
    -------
    Bn : array
        dipole field in northward direction, in nT. Same shape as mlat/r
    Br : array
        dipole field in radial direction, in nT. Same shape as mlat/r
    """

    shape = np.broadcast(mlat, r).shape
    colat = (90 - (mlat * np.ones_like(r)).flatten()) * d2r
    r    = (np.ones_like(mlat) * r).flatten()


    # Find IGRF parameters for given epoch:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + [epoch]).sort_index().interpolate().drop_duplicates() 
    dipole = dipole.loc[epoch, :]

    B0 = dipole['B0']

    Bn = B0 * (RE / r) ** 3 * np.sin( colat )
    Br = -2 * B0 * (RE / r) ** 3 * np.cos( colat )

    return Bn.reshape(shape), Br.reshape(shape)




def dipole_axis(epoch):
    """ calculate dipole axis in geocentric ECEF coordinates for given epoch(s)

    Calculations are based on IGRF coefficients, and linear interpolation is used 
    in between IGRF models (defined every 5 years). Secular variation coefficients
    are used for the five years after the latest model. 

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value

    Returns
    -------
    axes : array
        N x 3-dimensional array, where N is the number of inputs (epochs), and the
        columns contain the x, y, and z components of the corresponding dipole axes

    """

    epoch = np.asarray(epoch).flatten() # turn input into array in case it isn't already

    # interpolate Gauss coefficients to the input times:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + list(epoch)).sort_index().interpolate().drop_duplicates() 

    params = {key: dipole.loc[epoch, key].values for key in ['g10', 'g11', 'h11', 'B0']}

    Z_cd = -np.vstack((params['g11'], params['h11'], params['g10']))/params['B0']

    return Z_cd.T




def dipole_poles(epoch):
    """ calculate dipole pole positions at given epoch(s)

    Parameters
    ----------
    epoch : float or array of floats
        year (with fraction) for which the dipole axis will be calculated. Multiple
        epochs can be given, as an array of N floats, resulting in a N x 3-dimensional
        return value
    
    Returns
    -------
    north_colat : array
        colatitude of the dipole pole in the northern hemisphere, same number of
        values as input
    north_longitude: array
        longitude of the dipole pole in the northern hemisphere, same number of
        values as input
    south_colat : array
        colatitude of the dipole pole in the southern hemisphere, same number of
        values as input
    south_longitude: array
        longitude of the dipole pole in the southern hemisphere, same number of
        values as input

       


    """
    print(dipole_axis(epoch))
    north_colat, north_longitude = car_to_sph( dipole_axis(epoch).T, deg = True)[1:]
    south_colat, south_longitude = car_to_sph(-dipole_axis(epoch).T, deg = True)[1:]
    
    return north_colat, north_longitude, south_colat, south_longitude


def geo2mag(glat, glon, Ae = None, An = None, epoch = 2020, deg = True, inverse = False):
    """ Convert geographic (geocentric) to centered dipole coordinates

    The conversion uses IGRF coefficients directly, interpolated
    to the provided epoch. The construction of the rotation matrix
    follows Laundal & Richmond (2017) [4]_ . 

    Preserves shape. glat, glon, Ae, and An should have matching shapes

    Parameters
    ----------
    glat : array_like
        array of geographic latitudes
    glon : array_like
        array of geographic longitudes
    Ae   : array-like, optional
        array of eastward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    An   : array-like, optional
        array of northtward vector components to be converted. Default
        is 'none', and no converted vector components will be returned
    epoch : float, optional
        epoch (year) for the dipole used in the conversion, default 2020
    deg : bool, optional
        True if input is in degrees, False otherwise
    inverse: bool, optional
        set to True to convert from magnetic to geographic. 
        Default is False

    Returns
    -------
    cdlat : ndarray
        array of centered dipole latitudes [degrees]
    cdlon : ndarray
        array of centered dipole longitudes [degrees]
    Ae_cd : ndarray
        array of eastward vector components in dipole coords
        (if Ae != None and An != None)
    An_cd : ndarray
        array of northward vector components in dipole coords
        (if Ae != None and An != None)

    """

    shape = np.asarray(glat).shape
    glat, glon = np.asarray(glat).flatten(), np.asarray(glon).flatten()

    # Find IGRF parameters for given epoch:
    dipole = igrf_dipole.reindex(list(igrf_dipole.index) + [epoch]).sort_index().interpolate().drop_duplicates() 
    dipole = dipole.loc[epoch, :]

    # make rotation matrix from geo to cd
    Zcd = -np.array([dipole.g11, dipole.h11, dipole.g10])/dipole.B0
    Zgeo_x_Zcd = np.cross(np.array([0, 0, 1]), Zcd)
    Ycd = Zgeo_x_Zcd / np.linalg.norm(Zgeo_x_Zcd)
    Xcd = np.cross(Ycd, Zcd)

    Rgeo_to_cd = np.vstack((Xcd, Ycd, Zcd))

    if inverse: # transpose rotation matrix to get inverse operation
        Rgeo_to_cd = Rgeo_to_cd.T

    # convert input to ECEF:
    colat = 90 - glat.flatten() if deg else np.pi/2 - glat.flatten()
    glon  = glon.flatten()
    r_geo = sph_to_car(np.vstack((np.ones_like(colat), colat, glon)), deg = deg)

    # rotate:
    r_cd = Rgeo_to_cd.dot(r_geo)

    # convert result back to spherical:
    _, colat_cd, lon_cd = car_to_sph(r_cd, deg = True)

    # return coords if vector components are not to be calculated
    if any([Ae is None, An is None]):
        return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape)

    Ae, An = np.asarray(Ae).flatten(), np.asarray(An).flatten()
    A_geo_enu  = np.vstack((Ae, An, np.zeros(Ae.size)))
    A = np.sqrt(Ae**2 + An**2)
    A_geo_ecef = enu_to_ecef((A_geo_enu / A).T, glon, glat ) # rotate normalized vectors to ecef
    A_cd_ecef = Rgeo_to_cd.dot(A_geo_ecef.T)
    A_cd_enu  = ecef_to_enu(A_cd_ecef.T, lon_cd, 90 - colat_cd).T * A 

    # return coords and vector components:
    return 90 - colat_cd.reshape(shape), lon_cd.reshape(shape), A_cd_enu[0].reshape(shape), A_cd_enu[1].reshape(shape)




