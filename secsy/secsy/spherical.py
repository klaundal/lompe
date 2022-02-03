""" Module for useful stuff when working in spherical coordinate system
"""

import numpy as np
d2r = np.pi/180
r2d = 180/np.pi


def sph_to_car(sph, deg = True):
    """ convert from spherical to cartesian coordinates

        input: 3 X N array:
           [r1    , r2    , ..., rN    ]
           [colat1, colat2, ..., colatN]
           [lon1  , lon2  , ..., lonN  ]

        output: 3 X N array:
           [x1, x2, ... xN]
           [y1, y2, ... yN]
           [z1, z2, ... zN]

        deg = True if lat and lon are given in degrees, 
              False if radians
    """

    r, theta, phi = sph

    if deg == False:
        conv = 1.
    else:
        conv = d2r


    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))

def car_to_sph(car, deg = True):
    """ convert from cartesian to spherical coordinates

        input: 3 X N array:
           [x1, x2, ... xN]
           [y1, y2, ... yN]
           [z1, z2, ... zN]

        output: 3 X N array:
           [r1    , r2    , ..., rN    ]
           [colat1, colat2, ..., colatN]
           [lon1  , lon2  , ..., lonN  ]

        deg = True if lat and lon is wanted in degrees
              False if radians
    """

    x, y, z = car

    if deg == False:
        conv = 1.
    else:
        conv = r2d

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)*conv
    phi = ((np.arctan2(y, x)*180/np.pi) % 360)/180*np.pi * conv

    return np.vstack((r, theta, phi))


def sph_to_sph(lat, lon, x_lat, x_lon, z_lat, z_lon, deg = True):
    """ calculate the latitude and longitude in a spherical coordinate system
        with the north pole at pole_lat, pole_lon. lat, lon are latitude and 
        longitude in the original coordinate system

        the cooridnates of the new z and x axes must be given. They must describe
        orthogonal positions, otherwise an exception is raised

        parameters
        ----------
        lat : array
            latitude of the points that will be converted - will be flattened
        lon : array
            longitude of the points that will be converted - will be flattened
        x_lat : float
            latitude of the new x axis
        x_lon : float
            longitude of the new x axis
        z_lat : float
            latitude of the new z axis
        z_lon : float
            longitude of the new z axis
        deg : bool, optional
            True if input and output in degrees, False if radians.
            Default is True

        output
        ------
        latitude and longitude in the new coordinate system. These are arrays with the same
        size as lat and lon (although, shape is not conserved if input dimensions are > 1)
    """
    lat, lon = lat.flatten(), lon.flatten()

    if deg == False:
        conv = 1.
    else:
        conv = d2r

    xyz = np.vstack((np.cos(lat * conv) * np.cos(lon * conv), 
                     np.cos(lat * conv) * np.sin(lon * conv), 
                     np.sin(lat * conv)))

    new_z = np.array([np.cos(z_lat * conv) * np.cos(z_lon * conv), 
                      np.cos(z_lat * conv) * np.sin(z_lon * conv),
                      np.sin(z_lat * conv)                          ])
    new_x = np.array([np.cos(x_lat * conv) * np.cos(x_lon * conv), 
                      np.cos(x_lat * conv) * np.sin(x_lon * conv),
                      np.sin(x_lat * conv)                          ])
    new_y = np.cross(new_z, new_x, axisa = 0, axisb = 0, axisc = 0)
    new_x, new_y, new_z = new_x.flatten(), new_y.flatten(), new_z.flatten()

    # if new_y is not a unit vector, new_x and new_z are not orthogonal:
    if not np.isclose(np.linalg.norm(new_y), 1):
        raise ValueError('x and z coords do not describe orthogonal positions')

    # make rotation matrix and do the rotation
    R = np.vstack((new_x, new_y, new_z))
    XYZ = R.dot(xyz)

    # convert back to spherical
    _, COLAT, LON = car_to_sph(XYZ, deg = deg)

    return 90 - COLAT, LON


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


def tangent_vector(lat1, lon1, lat2, lon2, degrees = True):
    """ calculate tangential (to a sphere) unit vector at (lat1, lon1) pointing towards (lat2, lon2) 

        input must be arrays with equal shape:
        lat1, lon1 -- latitude (not colat) and longitude of origin
        lat2, lon2 -- latitude (not colat) and longitude which return vector points towards
        degrees    -- True if input in degrees, False if radians

        output:
        east, north -- eastward and northward components of tangential unit vector

        Will raise ValueError if
          - inputs do not have equal shapes
          - inputs contain points that are closer to identical or antipodal than (roughly) 0.3 degrees

        vectorized code (fast)

        KML 2016-04-20
        2020-04 - fixed check to see if tangent is well defined
    """

    if not (lat1.shape == lon1.shape == lat2.shape == lon2.shape):
        raise ValueError('tangent_vector: input coordinates do not have equal shapes')

    shape = lat1.shape

    # convert to radians if necessary, and flatten:
    if degrees:
        converter = lambda x: x.flatten() * np.pi/180.
    else:
        converter = lambda x: x.flatten()

    lat1, lon1, lat2, lon2 = list(map(converter, (lat1, lon1, lat2, lon2)))

    # ECEF position vectors:
    ecef_p1 = np.vstack( (np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)) )
    ecef_p2 = np.vstack( (np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)) )

    # check if tangent is well defined:
    if np.any(np.isclose(    np.sum((ecef_p1*ecef_p2)**2, axis = 0), 1.)):
        points = np.isclose( np.sum((ecef_p1*ecef_p2)**2, axis = 0), 1.).nonzero()[0]
        raise ValueError('tangent_vector: input coordinates at nearly identical or antipodal points; tangent not defined\n flattened coordinates: %s' % points)

    # non-tangential difference vector (3, N):
    dp = ecef_p2 - ecef_p1

    # subtract normal part of the vectors to make tangential vector in ECEF coordinates:
    ecef_t = dp - np.sum(dp * ecef_p1, axis = 0) * ecef_p1

    # normalize:
    ecef_t = ecef_t / np.linalg.norm(ecef_t, axis = 0)

    # convert ecef_t to enu_t, by constructing N rotation matrices:
    R = np.dstack( (np.vstack((-np.sin(lon1)                ,  np.cos(lon1)               ,    0 * lat1 )).T,
                    np.vstack((-np.cos(lon1)  * np.sin(lat1), -np.sin(lon1) * np.sin(lat1), np.cos(lat1))).T,
                    np.vstack(( np.cos(lon1)  * np.cos(lat1),  np.sin(lon1) * np.cos(lat1), np.sin(lat1))).T))

    enu_t = np.einsum('lji, jl->il', R, ecef_t)
    enu_t = enu_t[:2] # third coordinate (up) is zero, since normal part was removed

    # extract east and north components, reshape to original shape and return stacked
    east = enu_t[0].reshape(shape)
    north = enu_t[1].reshape(shape)

    return east, north




if __name__ == '__main__':

    # TESTING ENU/ECEF CONVERSION:
    v = np.array([[1, 1, 0], [1, 0, 0]])
    lat = np.array([-90, 0])
    lon = np.array([0., 0])
    print(enu_to_ecef(v, lat, lon))

    v = (np.random.random((30, 3)) - .5)*300
    lat = (np.random.random(30) - .5) * 180
    lon = np.random.random(30) * 360
    print('This number should be small: ', np.max(enu_to_ecef(enu_to_ecef(v, lat, lon), lat, lon, reverse = True) - v)**2)
