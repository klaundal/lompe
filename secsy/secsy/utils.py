""" 
SECS utils

"""

import numpy as np
d2r = np.pi/180
MU0 = 4 * np.pi * 1e-7
RE = 6371.2 * 1e3


def dpclip(x, delta = 1e-7):
    """ 
    dot product clip:
    clip x to values between -1 + delta and 1 - delta
    """
    return np.clip(x, -1 + delta, 1 - delta)

def get_theta(lat, lon, lat_secs, lon_secs, return_degrees = False):
    """" calculate theta angle - the angle between data point and secs node.

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
        Output will be a 2D array with shape (mlat.size, mlat_secs.size)
    return_degrees: bool, optional
        Set to True if you want output in degrees. Default is False (radians)

    Returns
    -------
    theta: 2D array (lat.size, lat_secs.size)
        Array of polar angles, angular distances between the points
        described by (lat, lon) and the points described by 
        (lat_secs, lon_secs). Unit in radians unless return_degrees is set
        to True
    """

    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T

    # the polar angles (N, M):
    theta = np.arccos(dpclip(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs)))

    if return_degrees:
        theta = theta / d2r 

    return theta


def get_SECS_J_G_matrices(lat, lon, lat_secs, lon_secs, 
                          current_type = 'divergence_free', constant = 1./(4*np.pi), 
                          RI = RE + 110 * 1e3,
                          singularity_limit = 0):
    """ Calculate matrices Ge and Gn which relate SECS amplitudes to current density 
        vector components.

    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    current_type: string, optional
        The type of SECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
        'potential': scalar field whose negative gradient is curl-free SECS
        'scalar': 
    constant: float, optional
        The SECS functions are scaled by the factor 1/(4pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m
    singularity_limit: float (optional)
        A modified version of the SECS functions will be used at
        points that are closer than singularity_limit. The modification
        is given by equations 2.43 (CF) and 2.44 (DF) in Vanhamaki and
        Juusola (2020), and singularity_limit / RI is equal to theta0
        in these equations. Default is 0, which means that the original
        version of the SECS functions are used (with singularities). 
        singularity_limit is ignored if current_type is 'potential' or 'scalar'

    Returns
    -------
    If current_type is 'divergence_free' or 'curl_free':
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward current densities at (lat, lon) via 'je = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward current densities at (lat, lon) via 'jn = Gn.dot(m)'
    If current_type is 'potential' or 'scalar':
    G: 2D array
        2D array with shape (lat.size, lat_secs.size), relating amplitudes m
        to scalar field magnitude at (lat, lon) via 'z = G.dot(m)'

    """
    
    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T
    
    # unit vector pointing from SECS to data points - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECS:
    R = np.hstack( (np.dstack((-np.sin(lo)              ,  np.cos(lo)             , np.zeros_like(la) )),
                    np.dstack((-np.cos(lo)  * np.sin(la), -np.sin(lo) * np.sin(la), np.cos(       la) )),
                    np.dstack(( np.cos(lo)  * np.cos(la),  np.sin(lo) * np.cos(la), np.sin(       la) ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECS
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion
    
    if current_type == 'divergence_free':
        # rotate these vectors to get vectors pointing eastward with respect to SECS systems at each data point
        enu_vec = np.dstack((enu_t[:, :, 1], -enu_t[:, :, 0])) # north -> east and east -> south
    elif current_type == 'curl_free':
        enu_vec = -enu_t # outward from SECS
    elif current_type in ['potential', 'scalar']:
        enu_vec = 1
    else:
        raise Exception('type must be "divergence_free", "curl_free", "potential", or "sclar"')

    # get the scalar part of Amm's divergence-free SECS:    
    theta  = np.arccos(dpclip(np.einsum('ij,kj->ik', ecef_r_secs, ecef_r_data)))
    if current_type in ['divergence_free', 'curl_free']:
        coeff = constant /np.tan(theta/2)/ RI

        # apply modifications to handle singularities:
        theta0 = singularity_limit / RI
        if theta0 > 0:
            alpha = 1 / np.tan(theta0/2)**2
            coeff[theta < theta0] = constant * alpha * np.tan(theta[theta < theta0]/2) / RI

        # G matrices
        Ge = coeff * enu_vec[:, :, 0].T
        Gn = coeff * enu_vec[:, :, 1].T
    
        return Ge.T, Gn.T
    else: # current_type is 'potential' or 'scalar'
        if current_type == 'potential':
            return -2*constant*np.log(np.sin(theta/2)).T
        elif current_type == 'scalar':
            return    constant      / np.tan(theta/2).T


def get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs,
                          current_type = 'divergence_free', constant = 1./(4*np.pi), 
                          RI = RE + 110 * 1e3,
                          singularity_limit = 0,
                          induction_nullification_radius = None):
    """ Calculate matrices Ge, Gn, and Gr which relate SECS amplitudes to magnetic field

    Based on equations (9) and (10) of Amm and Viljanen 1999, or (2.13)-(2.14) in Vanhamaki
    and Juusola 2020. 

    If singularity_limit > 0, the magnetic field of curl-free currents is modified, but
    not the magnetic field of divergence-free currents (!). See Section 2.10.2 and 
    equation (2.46) in Vanhamaki and Juusola 2020.


    Parameters
    ----------
    lat: array-like
        Array of latitudes of evaluation points [deg]
        Flattened array must have same size as lon
    lon: array-like
        Array of longitudes of evaluation points [deg].
        Flattened array must have same size as lat
    r: array-like
        Array of radii of evaluation points. Flattened
        array must either have size 1, in which case one
        radius is used for all points, or have same size as 
        lat. Unit should be the same as RI 
    lat_secs: array-like
        Array of SECS pole latitudes [deg]
        Flattened array must have same size as lon_secs
    lon_secs: array-like
        Array of SECS pole longitudes [deg]
        Flattened array must havef same size as lat_secs
    current_type: string, optional
        The type of SECS function. This must be either 
        'divergence_free' (default): divergence-free basis functions
        'curl_free': curl-free basis functions
    constant: float, optional
        The SECS functions are scaled by the factor 1/(4pi), which is
        the default value of 'constant'. Change if you want something 
        different.
    RI: float (optional)
        Radius of SECS poles. Default is Earth radius + 110,000 m
    singularity_limit: float (optional)
        A modified version of the SECS functions will be used at
        points that are closer than singularity_limit. The modification
        is given by equations 2.43 (CF) and 2.44 (DF) in Vanhamaki and
        Juusola (2020), and singularity_limit / RI is equal to theta0
        in these equations. Default is 0, which means that the original
        version of the SECS functions are used (with singularities). 
    induction_nullification_radius: float or None, optional
        The radius at which ground induced image currents cancel the radial
        magnetic field. Default in None, in which case there are no
        induced image currents. This part is based on equations of Appendix A
        in "Juusola, L., Kauristie, K., Vanhamäki, H., Aikio, A., and 
        van de Kamp, M. (2016), Comparison of auroral ionospheric and field‐
        aligned currents derived from Swarm and ground magnetic field measurements, 
        J. Geophys. Res. Space Physics, 121, 9256– 9283, doi:10.1002/2016JA022961."

    Returns
    -------
    Ge: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the eastward magnetic field at (lat, lon) via 'Be = Ge.dot(m)'
    Gn: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the northward magnetic field at (lat, lon) via 'Bn = Gn.dot(m)'
    Gr: 2D array
        2D array with shape (lat.size, lat_secs.size), relating SECS amplitudes
        m to the radial magnetic field at (lat, lon) via 'Br = Gr.dot(m)'

    """
    
    # reshape angles and convert to radians:
    la   = np.array(lat).flatten()[:, np.newaxis] * d2r
    lo   = np.array(lon).flatten()[:, np.newaxis] * d2r
    la_s = np.array(lat_secs).flatten()[np.newaxis, :] * d2r
    lo_s = np.array(lon_secs).flatten()[np.newaxis, :] * d2r

    # reshape r:
    if np.array(r).size == 1:
        r = np.ones_like(la) * r
    else:
        r = np.array(r).flatten()[:, np.newaxis]

    # ECEF position vectors of data points - should be N by 3, where N is number of data points
    ecef_r_data = np.hstack((np.cos(la  ) * np.cos(lo  ), np.cos(la  ) * np.sin(lo  ), np.sin(la  )))
    
    # position vectors SECS poles - should be 3 by M, where M is number of SECS - these are the z axes of each SECS 
    ecef_r_secs = np.vstack((np.cos(la_s) * np.cos(lo_s), np.cos(la_s) * np.sin(lo_s), np.sin(la_s))).T
    
    # unit vector pointing from SECS to data points - (M, N, 3) 
    ecef_t = ecef_r_secs[np.newaxis, :, :] - ecef_r_data[:, np.newaxis, :] # difference vector - not tangential yet
    ecef_t = ecef_t - np.einsum('ijk,ik->ij', ecef_t, ecef_r_data)[:, :, np.newaxis] * ecef_r_data[:, np.newaxis, :] # subtract radial part of the vector to make it tangential
    ecef_t = ecef_t/np.linalg.norm(ecef_t, axis = 2)[:, :, np.newaxis] # normalize the result
        
    # make N rotation matrices to rotate ecef_t to enu_t - one rotation matrix per SECS:
    R = np.hstack( (np.dstack((-np.sin(lo)              ,  np.cos(lo)             , np.zeros_like(la) )),
                    np.dstack((-np.cos(lo)  * np.sin(la), -np.sin(lo) * np.sin(la), np.cos(       la) )),
                    np.dstack(( np.cos(lo)  * np.cos(la),  np.sin(lo) * np.cos(la), np.sin(       la) ))) )

    # apply rotation matrices to make enu vectors pointing from data points to SECS
    enu_t = np.einsum('lij, lkj->lki', R, ecef_t)[:, :, :-1] # remove last component (up), which should deviate from zero only by machine precicion

    # the polar angles (N, M):
    theta = np.arccos(dpclip(np.einsum('ij, kj -> ik', ecef_r_data, ecef_r_secs)))


    # indices of data points that are below and above current sheet:
    below = r.flatten() <= RI
    above = r.flatten()  > RI

    # G matrix scale factors
    if current_type == 'divergence_free':
        s = np.minimum(r, RI) / np.maximum(r, RI)
        sa = s[above]
        sb = s[below]

        Ar  = MU0 * constant /  r                   # common factor radial direction
        Sr = np.zeros_like(theta)
        Sr[below] = 1  / np.sqrt(1 + sb**2 - 2 * sb * np.cos(theta[below])) - 1
        Sr[above] = sa / np.sqrt(1 + sa**2 - 2 * sa * np.cos(theta[above])) - sa
        Gr = Ar * Sr

        An_ = MU0 * constant / (r * np.sin(theta))  # common factor local northward (note sign difference wrt theta) direction
        Sn_ = np.zeros_like(theta)
        Sn_[below] = (sb -      np.cos(theta[below])) / np.sqrt(1 + sb**2 - 2 * sb * np.cos(theta[below])) + np.cos(theta[below])
        Sn_[above] = (1  - sa * np.cos(theta[above])) / np.sqrt(1 + sa**2 - 2 * sa * np.cos(theta[above])) - 1
        Gn_ = An_ * Sn_        

        # calculate geo east, north:
        Ge =  Gn_ * enu_t[:, :, 0]
        Gn =  Gn_ * enu_t[:, :, 1]

    elif current_type == 'curl_free':
        # G matrix for local eastward component
        Ge_ = -MU0 * constant / np.tan(theta/2) / r 

        # apply modifications to handle singularities:
        theta0 = singularity_limit / RI
        if theta0 > 0:
            alpha = 1 / np.tan(theta0/2)**2
            rr = np.tile(r, (1, theta.shape[1])) # need one r for every element in matrix
            Ge_[theta < theta0] = -MU0 * constant * alpha * np.tan(theta[theta < theta0]/2) / rr[theta < theta0]

        # zero below current sheet:
        Ge_[below] *= 0 

        # calculate geo east, north, radial:
        Ge =  Ge_ * enu_t[:, :, 1] # eastward component of enu_t is northward component of enu_e (unit vector in local east direction)
        Gn = -Ge_ * enu_t[:, :, 0] # northward component of enu_t is eastward component of enu_e
        Gr =  Ge_ * 0              # no radial component


    if induction_nullification_radius != None and current_type == 'divergence_free':
        # include the effect of telluric image currents
        radius = induction_nullification_radius**2 / RI
        amplitude_factor = -RI / induction_nullification_radius

        Ge_, Gn_, Gr_ = get_SECS_B_G_matrices(lat, lon, r, lat_secs, lon_secs, 
                                              current_type = 'divergence_free',
                                              RI = radius)
        Ge = Ge + amplitude_factor * Ge_
        Gn = Gn + amplitude_factor * Gn_
        Gr = Gr + amplitude_factor * Gr_



    return Ge, Gn, Gr



