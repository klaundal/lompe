""" functions to calculate the geometry of Sunlight and the meaning of life """

import pandas as pd
import os
import numpy as np
from lompe.utils.time import date_to_doy, is_leapyear
from lompe.utils.coords import sph_to_car, car_to_sph


def subsol(datetimes):
    """ 
    calculate subsolar point at given datetime(s)

    returns:
      subsol_lat  -- latitude(s) of the subsolar point
      subsol_lon  -- longiutde(s) of the subsolar point

    The code is vectorized, so it should be fast.

    After Fortran code by: 961026 A. D. Richmond, NCAR

    Documentation from original code:
    Find subsolar geographic latitude and longitude from date and time.
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994).
    Usable for years 1601-2100, inclusive.  According to the Almanac, 
    results are good to at least 0.01 degree latitude and 0.025 degree 
    longitude between years 1950 and 2050.  Accuracy for other years 
    has not been tested.  Every day is assumed to have exactly
    86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored:  their effect is below the accuracy threshold of
    the algorithm.

    Added by SMH 2020/04/03 (from Kalle's code stores!)
    """

    # use pandas DatetimeIndex for fast access to year, month day etc...
    if hasattr(datetimes, '__iter__'): 
        datetimes = pd.DatetimeIndex(datetimes)
    else:
        datetimes = pd.DatetimeIndex([datetimes])

    year = np.array(datetimes.year)
    # day of year:
    doy  = date_to_doy(datetimes.month, datetimes.day, is_leapyear(year))
    # seconds since start of day:
    ut   = datetimes.hour * 60.**2 + datetimes.minute*60. + datetimes.second 
 
    yr = year - 2000

    if year.max() >= 2100 or year.min() <= 1600:
        raise ValueError('subsol.py: subsol invalid after 2100 and before 1600')

    nleap = np.floor((year-1601)/4.)
    nleap = nleap - 99

    # exception for years <= 1900:
    ncent = np.floor((year-1601)/100.)
    ncent = 3 - ncent
    nleap[year <= 1900] = nleap[year <= 1900] + ncent[year <= 1900]

    l0 = -79.549 + (-.238699*(yr-4*nleap) + 3.08514e-2*nleap)

    g0 = -2.472 + (-.2558905*(yr-4*nleap) - 3.79617e-2*nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut/86400. - 1.5) + doy

    # Addition to Mean longitude of Sun since January 1 of IYR:
    lf = .9856474*df

    # Addition to Mean anomaly since January 1 of IYR:
    gf = .9856003*df

    # Mean longitude of Sun:
    l = l0 + lf

    # Mean anomaly:
    g = g0 + gf
    grad = g*np.pi/180.

    # Ecliptic longitude:
    lmbda = l + 1.915*np.sin(grad) + .020*np.sin(2.*grad)
    lmrad = lmbda*np.pi/180.
    sinlm = np.sin(lmrad)

    # Days (including fraction) since 12 UT on January 1 of 2000:
    n = df + 365.*yr + nleap

    # Obliquity of ecliptic:
    epsilon = 23.439 - 4.e-7*n
    epsrad  = epsilon*np.pi/180.

    # Right ascension:
    alpha = np.arctan2(np.cos(epsrad)*sinlm, np.cos(lmrad)) * 180./np.pi

    # Declination:
    delta = np.arcsin(np.sin(epsrad)*sinlm) * 180./np.pi

    # Subsolar latitude:
    sbsllat = delta

    # Equation of time (degrees):
    etdeg = l - alpha
    nrot = np.round(etdeg/360.)
    etdeg = etdeg - 360.*nrot

    # Apparent time (degrees):
    aptime = ut/240. + etdeg    # Earth rotates one degree every 240 s.

    # Subsolar longitude:
    sbsllon = 180. - aptime
    nrot = np.round(sbsllon/360.)
    sbsllon = sbsllon - 360.*nrot

    return sbsllat, sbsllon


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
        conv = 180/np.pi
    else:
        conv = 1.

    # compute subsolar point
    sslat, sslon = subsol(datetimes)

    # compute and return the angle
    ssr = sph_to_car(np.vstack((np.ones_like(sslat), 90. - sslat, sslon)), deg = True)
    gcr = sph_to_car(np.vstack((np.ones_like(glat ), 90. - glat , glon )), deg = True)

    # the angle is arccos of the dot product of these two vectors
    return np.arccos(np.sum(ssr*gcr, axis = 0))*conv



def terminator(datetime, sza = 90, resolution = 360):
    """ compute terminator trajectory (constant solar zenith angle contour)

        glat, glon = compute_terminator(date, sza = 90, resolution = 360)

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
    sslon = sslon[0]*np.pi/180
    sslat = sslat[0]*np.pi/180

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
    rotation_angle = -(sza - 90) * np.pi/180

    sza_vector = sza90 * np.cos(rotation_angle) + np.cross(t0, sza90) * np.sin(rotation_angle) + t0 * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle)) # (rodrigues formula)
    sza_vector = sza_vector.flatten()

    # rotate this about the sun-Earth line to trace out the trajectory:
    angles = np.r_[0 : 2*np.pi: 2*np.pi / resolution][np.newaxis, :]
    r = sza_vector[:, np.newaxis] * np.cos(angles) + np.cross(ss, sza_vector)[:, np.newaxis] * np.sin(angles) + ss[:, np.newaxis] * (np.sum(t0*sza90)) * (1 - np.cos(rotation_angle))

    # convert to spherical and return
    tsph = car_to_sph(r, deg = True)

    return 90 - tsph[1], tsph[2]
