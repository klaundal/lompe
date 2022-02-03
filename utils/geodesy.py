""" functions for conversion between geocentric and geodetic coordinates """

import numpy as np

d2r = np.pi/180
WGS84_e2 = 0.00669437999014
WGS84_a  = 6378.137


def geod2geoc(gdlat, height, X, Z):
    """
    theta, r, B_th, B_r = geod2lat(gdlat, height, X, Z)

       INPUTS:    
       gdlat is geodetic latitude (not colat)
       height is geodetic height (km)
       X is northward vector component in geodetic coordinates 
       Z is downward vector component in geodetic coordinates

       OUTPUTS:
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_th is geocentric southward component (theta direction)
       B_r is geocentric radial component


    after Matlab code by Nils Olsen, DTU
    """

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    sin_alpha_2 = np.sin(gdlat*d2r)**2
    cos_alpha_2 = np.cos(gdlat*d2r)**2

    # calculate geocentric latitude and radius
    tmp = height * np.sqrt(a**2 * cos_alpha_2 + b**2 * sin_alpha_2)
    beta = np.arctan((tmp + b**2)/(tmp + a**2) * np.tan(gdlat * d2r))
    theta = np.pi/2 - beta
    r = np.sqrt(height**2 + 2 * tmp + a**2 * (1 - (1 - (b/a)**4) * sin_alpha_2) / (1 - (1 - (b/a)**2) * sin_alpha_2))

    # calculate geocentric components
    psi  =  np.sin(gdlat*d2r) * np.sin(theta) - np.cos(gdlat*d2r) * np.cos(theta)
    
    B_r  = -np.sin(psi) * X - np.cos(psi) * Z
    B_th = -np.cos(psi) * X + np.sin(psi) * Z

    theta = theta/d2r

    return theta, r, B_th, B_r
 

def geoc2geod(theta, r, B_th, B_r):
    """
    gdlat, height, X, Z = geod2lat(theta, r, B_th, B_r)

       INPUTS:    
       theta is geocentric colatitude (degrees)
       r is geocentric radius (km)
       B_r is geocentric radial component
       B_th is geocentric southward component (theta direction)

       OUTPUTS:
       gdlat is geodetic latitude (degrees, not colat)
       height is geodetic height (km)
       X is northward vector component in geodetic coordinates 
       Z is downward vector component in geodetic coordinates


    after Matlab code by Nils Olsen, DTU
    """
    
    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    OME2REQ = (1.-E2)*a
    A21 =     (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 =     (                        E6 +     E8)/  32.
    A23 = -3.*(                     4.*E6 +  3.*E8)/ 256.
    A41 =    -(           64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 =     (            4.*E4 +  2.*E6 +     E8)/  16.
    A43 =                                   15.*E8 / 256.
    A44 =                                      -E8 /  16.
    A61 =  3.*(                     4.*E6 +  5.*E8)/1024.
    A62 = -3.*(                        E6 +     E8)/  32.
    A63 = 35.*(                     4.*E6 +  3.*E8)/ 768.
    A81 =                                   -5.*E8 /2048.
    A82 =                                   64.*E8 /2048.
    A83 =                                 -252.*E8 /2048.
    A84 =                                  320.*E8 /2048.
    
    GCLAT = (90-theta)
    SCL = np.sin(GCLAT * d2r)
    
    RI = a/r
    A2 = RI*(A21 + RI * (A22 + RI* A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI* A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))
    
    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL  * CCL
    C2CL = 2.*CCL  * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL
    
    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + GCLAT * d2r
    height = r * np.cos(DLTCL)- a * np.sqrt(1 -  E2 * np.sin(gdlat) ** 2)


    # magnetic components 
    psi = np.sin(gdlat) * np.sin(theta*d2r) - np.cos(gdlat) * np.cos(theta*d2r)
    X  = -np.cos(psi) * B_th - np.sin(psi) * B_r 
    Z  =  np.sin(psi) * B_th - np.cos(psi) * B_r 

    gdlat = gdlat / d2r

    return gdlat, height, X, Z
