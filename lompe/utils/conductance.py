""" Conductance functions"""

import numpy as np
import pandas as pd
import os
import apexpy
from scipy.interpolate import interp1d
from lompe.utils import sunlight
from lompe.dipole.dipole import Dipole
from scipy.interpolate import RectBivariateSpline


d2r = np.pi/180

def hardy_EUV(lon, lat, kp, time, hall_or_pedersen ='hp', starlight = 0, F107 = 100,
              dipole=False, calibration = 'MoenBrekke1993'):
    """ calculate conductance at lat, lon for given kp at given time
    based on Hardy model + EUV contribution, from the functions defined below
    

    Parameters
    ----------
    lon: array
        geographic longitudes [deg]
    lat: array
        geograhpic latitudes [deg]
    kp: int
        Kp index (used for Hardy model)
    time: datetime
        time, used to get solar zenith angles and for apex coordinate conversion
    hall_or_pedersen: string
        specifies type of conductance, 'hall' or 'pedersen'
    starlight: float, optional
        constant to add to conductance, often small (e.g. Strobel et al. 1980 https://doi.org/10.1016/0032-0633(80)90050-1)
        could be used to include "background-conductance" 
    F107: float, optional
        F107 index - used to scale EUV conductance. Default is 100
    dipole : bool, optional
        set to True if lat and lon are dipole coordinates. Default is False
    calibration: string, optional
        calibration to use in EUV_conductance calculation. See documentation
        of EUV_conductance function for info
        
    Returns
    -------
    If hall_or_pedersen == 'hall':
        Total Hall conductances [mho] for each lat, lon
    If hall_or_pedersen == 'pedersen':
        Total Pedersen conductances [mho] for each lat, lon
    If hall_or_pedersen == 'hallandpedersen' or 'hp':
        Two arrays of conductances [mho] for each lat, lon, 
        one for total Hall and one for total Pedersen 
    
    """
    assert hall_or_pedersen.lower() in ['hall', 'h', 'pedersen', 'p', 'hp', 'hallandpedersen'], "hardy_EUV: hall_or_pedersen must be either hall or pedersen, or hallandpedersen"
    
    lat, lon = np.array(lat, ndmin = 1), np.array(lon, ndmin = 1)
    shape = np.broadcast(lat, lon).shape
    lat, lon = lat.flatten(), lon.flatten()
    
    cd = Dipole(time.year)       
    if dipole:
        mlat, mlon = lat, lon # input lat, lon is centered dipole
        lat, lon = cd.mag2geo(lat, lon) # to geographic
    else:
        a = apexpy.Apex(time, 110) 
        mlat, mlon = a.geo2apex(lat, lon, 110) # to mag
    mlt = cd.mlon2mlt(mlon, time)     # get mlt
    
    # solar zenith angles for EUV conductances
    sza = sunlight.sza(lat, lon, time)
        
    if hall_or_pedersen.lower() in 'hall':
        hop = 'h'
    if hall_or_pedersen.lower() in 'pedersen':
        hop = 'p'
    if hall_or_pedersen.lower() in ['hp', 'hallandpedersen']:
        hop = 'hp'
    
    if len(hop) > 1:
        EUVh, EUVp = EUV_conductance(sza, F107, hop, calibration = calibration) # EUV
        hc_hall, hc_pedersen = hardy(mlat, mlt, kp, hop)                        # auroral
    else:
        EUV = EUV_conductance(sza, F107, hop, calibration = calibration)  # EUV
        hc  = hardy(mlat, mlt, kp, hop)                                   # auroral
    
    if hop == 'h':
        return (np.sqrt(hc**2 + EUV**2 + starlight**2)).reshape(shape)
    elif hop == 'p':
        return (np.sqrt(hc**2 + EUV**2 + starlight**2)).reshape(shape)
    else:
        return (np.sqrt(hc_hall**2 + EUVh**2 + starlight**2)).reshape(shape), (np.sqrt(hc_pedersen**2 + EUVp**2 + starlight**2)).reshape(shape)



def EUV_conductance(sza, F107 = 100, hallOrPed = 'hp',
                    calibration = 'MoenBrekke1993'):
    """
    cond = EUV_conductance(sza, F107, hallOrPed, calibration='MoenBrekke1993')

    Conductance calculated based on the plasma production at the height of max plasma production
    using the Chapman function (which assumes the earth is round, not flat) - and scaled to fit 
    your preferred empirical conductance model.

    The advantage of this approach is that the conductance gradient at the terminator is
    more realistic. This is important since conductance gradients appear in the equations that
    relate electric and magnetic fields. In addition, conductances above 90° sza are positive.

    The code is based on table lookup, and does not calculate the Chapman function.

    Author: S. M. Hatch (2021)

    Parameters
    ----------
    sza: 		array
        Solar zenith angle in degrees
    F107: float or array, optional
        F10.7 index - used to scale EUV conductance
        defualt is 100
    hallOrPed: 	string, optional
        Must be one of 'h', 'p', or 'hp', (corresponding to "Hall," "Pedersen," or both)
        default is both
    calibration: string, optional
        calibration to use in EUV_conductance calculation. Should be one of 
        'MoenBrekke1993', 'MoenBrekke1993_alt', 'Cousinsetal2015', with reference to
        the two papers cited below under References. 

        For Cousins et al (2015), Hall and Pedersen conductance are modeled using their 
        Equations (13) and (14).
        For Moen and Brekke (1993), Hall and Pedersen conductance are modeled using their 
        Equation (6).

    Returns
    -------
    If hall_or_pedersen == 'h':
        Hall conductances [mho] for each sza input value
    If hall_or_pedersen == 'p':
        Pedersen conductances [mho] for each sza input value
    If hall_or_pedersen == 'hp':
        Tuple of two arrays, one for Hall and one for Pedersen conductances [mho], for each sza input value
    

    Example
    -------
    # Get Hall conductance
    F107 = 70
    sza = np.arange(0,120.1,0.1)
    hall = EUV_conductance(sza,F107,'h')

    # Get Pedersen conductance
    F107 = 70
    sza = np.arange(0,120.1,0.1)
    pedersen = EUV_conductance(sza,F107,'p')

    # Get Hall and Pedersen conductance
    F107 = 70
    sza = np.arange(0,120.1,0.1)
    hall, pedersen = EUV_conductance(sza,F107,'hp')


    References
    ----------
    Cousins, E. D. P., Matsuo, T. and Richmond, A. D. (2015) ‘Mapping high-latitude ionospheric 
    electrodynamics with SuperDARN and AMPERE’, Journal of Geophysical Research: Space Physics, 
    120, pp. 5854–5870. doi: 10.1002/2014JA020463.

    Moen, J. and Brekke, A. (1993) ‘The solar flux influence on quiet time conductances in the 
    auroral ionosphere’, Geophysical Research Letters, 20(10), pp. 971–974. doi: 10.1029/92GL02109.
    
    """
    shape = np.array(sza).shape

    assert hallOrPed.lower() in ['h','p','hp'],"EUV_conductance: Must select one of 'h', 'p', or 'hp' for hallOrPed!"


    PRODUCTIONFILE = os.path.join(os.path.dirname(__file__), '../data/chapman_euv_productionvalues.txt')

    MODELSZAS = np.arange(0,120.1,0.1)  # DO NOT CHANGE
    PRODUCTION = np.loadtxt(PRODUCTIONFILE)

    getH = 'h' in hallOrPed.lower()
    getP = 'p' in hallOrPed.lower()

    if calibration not in ['MoenBrekke1993','MoenBrekke1993_alt','Cousinsetal2015']:
        defcal = 'MoenBrekke1993'
        print(f"Invalid calibration: {calibration}. Using {defcal}")
        calibration = defcal

    if calibration == 'MoenBrekke1993':
        # PedScl = 1.27               # Obtained by setting f107 = 1, sza = 0° in Moen and Brekke's (1993) empirical Pedersen conductance formula
        # HalScl = 1.35               # Obtained by setting f107 = 1, sza = 0° in Moen and Brekke's (1993) empirical Hall conductance formula

        f107pedexponent = 0.49
        f107hallexponent = 0.53

    elif calibration == 'MoenBrekke1993_alt':
        PedScl = 1.27               # Obtained by setting f107 = 1, sza = 0° in Moen and Brekke's (1993) empirical Pedersen conductance formula
        HalScl = 1.35               # Obtained by setting f107 = 1, sza = 0° in Moen and Brekke's (1993) empirical Hall conductance formula

        f107pedexponent = 0.49
        f107hallexponent = 0.53
        pedexponent=0.65
        hallexponent=0.79

    elif calibration == 'Cousinsetal2015':
        PedScl = 0.5               # Obtained by setting f107 = 1, sza = 0° in Brekke and Moen's (1993) empirical Pedersen conductance formula
        HalScl = 1.8               # Obtained by setting f107 = 1, sza = 0° in Brekke and Moen's (1993) empirical Hall conductance formula

        f107pedexponent = 0.667
        f107hallexponent = 0.5
        pedexponent=0.667
        hallexponent=1

    if calibration == 'MoenBrekke1993':

        if getH:
            halinterp = interp1d(MODELSZAS,
                                 F107**(f107hallexponent)*(0.81*PRODUCTION + 0.54*np.sqrt(PRODUCTION)),
                                 fill_value='extrapolate')
            sigh = halinterp(sza) # moh

        if getP:
            pedinterp = interp1d(MODELSZAS,
                                 F107**(f107pedexponent)*(0.34*PRODUCTION + 0.93*np.sqrt(PRODUCTION)),
                                 fill_value='extrapolate')
            sigp = pedinterp(sza) # moh

    else:

        if getH:
            halinterp = interp1d(MODELSZAS,
                                 F107**(f107hallexponent)*HalScl*(PRODUCTION)**(hallexponent),
                                 fill_value='extrapolate')
            sigh = halinterp(sza) # moh

        if getP:
            pedinterp = interp1d(MODELSZAS,
                                 F107**(f107pedexponent)*PedScl*(PRODUCTION)**(pedexponent),
                                 fill_value='extrapolate')
            sigp = pedinterp(sza) # moh


    if getH and getP:
        sigh[sigh < 0] = 0
        sigp[sigp < 0] = 0
        return sigh.reshape(shape), sigp.reshape(shape)
    elif getH:
        sigh[sigh < 0] = 0
        return sigh.reshape(shape)
    elif getP:
        sigp[sigp < 0] = 0
        return sigp.reshape(shape)


def hardy(mlat, mlt, kp, hallOrPed = 'hp'):
    """ calculte Hardy conductivity at given mlat/mlt, for given Kp 
    
    The model is described and defined in:
    Hardy, D. A., Gussenhoven, M. S., Raistrick, R., and McNeil, W. J. (1987), 
    Statistical and functional representations of the pattern of auroral energy 
    flux, number flux, and conductivity, J. Geophys. Res., 92( A11), 12275– 12294, 
    doi:10.1029/JA092iA11p12275.

    Parameters
    ----------
    mlat: array
        magnetic latitude in degrees
    mlt: array
        magnetic local time in hours
    kp: int
        Kp level, must be in [0, 1, ... 6]
    hallOrPed: str, optional
        Must be one of 'h', 'p', or 'hp', (corresponding to "Hall," "Pedersen," or both)
        default is both
    
    Returns
    -------
    Hall conductance: array
        array of Hall conductance [mho] with shape implied by mlat and mlt
    Peddersen conductance: array
        array of Pedersen conductance [mho] with shape implied by mlat and mlt
    
    if hallOrped == 'h':
        Hall conductance
    if hallOrped == 'p':
        Pedersen conductance
    if hallOrped == 'hp':
        Hall conductance, Pedersen conductance

    """
    assert hallOrPed.lower() in ['h','p','hp'], "hardy: Must select one of 'h', 'p', or 'hp' for hallOrPed!"
    assert kp in [0, 1, 2, 3, 4, 5, 6], "hardy: Kp must be an integer in the range 0-6"
    
    mlat, mlt = np.array(np.abs(mlat), ndmin = 1), np.array(mlt, ndmin = 1)
    shape = np.broadcast(mlat, mlt).shape
    mlat, mlt = mlat.flatten(), mlt.flatten()

    # load hall and pedersen coefficient files:
    basepath = os.path.dirname(__file__)
    
    # Hardy for Hall
    if 'h' in hallOrPed.lower():
        hc = pd.read_table(basepath + '/../data/hardy_hall_coefficients.txt'    , sep = ',', skipinitialspace=True, skiprows = [0,])
        hc = hc[hc.Kp == 'K' + str(kp)]     # select only relevant kp
        
        # define a dictionary whose keys are the 'term' column values, and values are the corresponding function of mlt
        hc['n']    = list(map(int, [t[-1] if t[-1] != 't' else 0 for t in hc['term']]))
        hc['trig'] = [np.sin if t[:3] == 'Sin' else np.cos  for t in hc['term']]
        
        # evaluate the fourier series        
        hall_epstein = dict(zip([u'maxvalue', u'maxlatitude', u'up-slope', u'down-slope'], [0]*4))
        for row in hc.iterrows():
            values = row[1]
            for key in hall_epstein:
                hall_epstein[key] += values[key] * values['trig'](values['n'] * mlt / 12 * np.pi)
        
        # evaluate the Epstein transition function, Hall:
        r, S1, S2, h0 = hall_epstein['maxvalue'], hall_epstein['up-slope'], hall_epstein['down-slope'], hall_epstein['maxlatitude']
        hall_conductance = r + S1*(mlat - h0) + (S2 - S1) * np.log((1 - S1/(S2 * np.exp(-(mlat - h0)))) / (1 - (S1/S2)))
        
        # introduce floors (using recommendation from paper)
        hall_conductance[    (mlat < h0) & (hall_conductance     < 0   )] = 0
        hall_conductance[    (mlat > h0) & (hall_conductance     < 0.55)] = 0.55
        
    # Hardy for Pedersen
    if 'p' in hallOrPed.lower():
        pc = pd.read_table(basepath + '/../data/hardy_pedersen_coefficients.txt', sep = ',', skipinitialspace=True, skiprows = [0,])
        pc = pc[pc.Kp == 'K' + str(kp)]     # select only relevant kp
        
        # define a dictionary whose keys are the 'term' column values, and values are the corresponding function of mlt
        pc['n']    = list(map(int, [t[-1] if t[-1] != 't' else 0 for t in pc['term']]))
        pc['trig'] = [np.sin if t[:3] == 'Sin' else np.cos  for t in pc['term']]    # the const term will be cos, but with n = 0
        
        # evaluate the fourier series
        pedersen_epstein = dict(zip([u'maxvalue', u'maxlatitude', u'up-slope', u'down-slope'], [0]*4))
        for row in pc.iterrows():
            values = row[1]
            for key in pedersen_epstein:
                pedersen_epstein[key] += values[key] * values['trig'](values['n'] * mlt / 12 * np.pi)    
    
        # evaluate the Epstein transition function, Pedersen:
        r, S1, S2, h0 = pedersen_epstein['maxvalue'], pedersen_epstein['up-slope'], pedersen_epstein['down-slope'], pedersen_epstein['maxlatitude']
        pedersen_conductance = r + S1*(mlat - h0) + (S2 - S1) * np.log((1 - S1/(S2 * np.exp(-(mlat - h0)))) / (1 - (S1/S2)))
        
        # introduce floors (using recommendation from paper)
        pedersen_conductance[(mlat < h0) & (pedersen_conductance < 0   )] = 0
        pedersen_conductance[(mlat > h0) & (pedersen_conductance < 0.55)] = 0.55
    
    if hallOrPed.lower() == 'h':
        return hall_conductance.reshape(shape)
    elif hallOrPed.lower() == 'p':
        return pedersen_conductance.reshape(shape)
    else:
        return hall_conductance.reshape(shape), pedersen_conductance.reshape(shape)


def calculate_robinson_conductance(FAC):
    ''' Estimate Hall and Pedersen conductance based on AMPERE FAC data, as
    described in 
    Robinson, R. M., Zanetti, L., Anderson, B., Vines, S., & Gjerloev, J. (2021). 
    Determination of auroral electrodynamic parameters from AMPERE field-aligned 
    current measurements. Space Weather, 19, e2020SW002677. 
    https://doi.org/10.1029/2020SW002677
    
    Parameters
    ----------
    FAC: array
        1200 element array of FAC values [muA/m2] downloaded from the AMPERE webpage
 
    
    Returns
    -------
    (sigmaH, sigmaP): 2 element tuple of 2D arrays
        array of Hall/Pedersen conductance [mho] with shape matching the 2D shape of 
        AMPERE FAC data
    
    '''

    #Arrays of the native AMPERE FAC product coordinates
    amp_sh = (24,50) # Shape of the AMPERE FAC data product
    amp_colat = np.tile(np.arange(1,amp_sh[1]+1),amp_sh[0]).reshape((amp_sh[0], 
                                                                     amp_sh[1]))
    amp_mlt = np.array([amp_sh[1]*[i] for i in np.arange(0,amp_sh[0])])
    amp_fac = FAC.copy().reshape(amp_sh)
    
    # load hall and pedersen coefficient file:
    basepath = os.path.dirname(__file__)
    coefs = pd.read_table(basepath + '/../data/robinson_conductance_coefs.txt', 
        skipinitialspace=True, skiprows = 1, delim_whitespace=True)
    # basepath = '/Users/jone/BCSS-DAG Dropbox/Jone Reistad/git/lompe/lompe'
    # coefs = pd.read_table(basepath + '/data/robinson_conductance_coefs.txt', 
        # skipinitialspace=True, skiprows = 1, delim_whitespace=True)
    
    #Apply eq4 in the paper
    ups = amp_fac >= 0
    downs = amp_fac < 0
    sigma_ = np.zeros(amp_sh)
    sigmaH = np.zeros(amp_sh)
    sigmaH[ups] = (sigma_+coefs.uh0.values[:,np.newaxis])[ups] + (sigma_+coefs.uh1.values[:,np.newaxis])[ups]*amp_fac[ups]
    sigmaH[downs] = (sigma_+coefs.dh0.values[:,np.newaxis])[downs] + (sigma_+coefs.dh1.values[:,np.newaxis])[downs]*amp_fac[downs]
    sigmaP = np.zeros(amp_sh)
    sigmaP[ups] = (sigma_+coefs.up0.values[:,np.newaxis])[ups] + (sigma_+coefs.up1.values[:,np.newaxis])[ups]*amp_fac[ups]
    sigmaP[downs] = (sigma_+coefs.dp0.values[:,np.newaxis])[downs] + (sigma_+coefs.dp1.values[:,np.newaxis])[downs]*amp_fac[downs]


    #Apply filtering when current density is low
    fill_value = 2 #mhos, from the paper
    rolling_  = pd.DataFrame(amp_fac.T).rolling(3, center=True).mean().T
    use = np.abs(rolling_) < 0.1
    sigmaP[use.values] = fill_value
    sigmaP[use.values] = fill_value
    
    #Filtering when signs are switching
    signed = pd.DataFrame(amp_fac)
    signed[ups] = 1
    signed[downs] = -1
    changed = np.diff(signed, axis=1, append=0)
    use = np.abs(changed) == 2 
    rolling_H  = pd.DataFrame(sigmaH.T).rolling(3, center=True).mean().fillna(method='bfill').fillna(method='ffill').T
    sigmaH[use] = rolling_H.values[use]
    rolling_P  = pd.DataFrame(sigmaP.T).rolling(3, center=True).mean().fillna(method='bfill').fillna(method='ffill').T
    sigmaP[use] = rolling_P.values[use]

    return (sigmaH, sigmaP)    


def robinson(mlat, mlt, sigmaH, sigmaP):
    ''' Interpolate value for conductance (from AMPERE FAC data product) onto 
    the location specified by mlat and mlt. 
    
    Parameters
    ----------
    mlat: array
        magnetic latitude in degrees
    mlt: array
        magnetic local time in hours
    sigmaH: 2D array
        Hall conductance [mho] on the native AMPERE grid.
    sigmaP: 2D array
        Pedersen conductance [mho] on the native AMPERE grid.         
    
    Returns
    -------
    (sigma_H, sigma_P): 2 element tuple of arrays
        Interpolated values of Hall and Pedersen conductances [mho] at the 
        user provided input locations mlat and mlt, with the same shape as input.
    '''
    #Checking input
    amp_sh = (24,50) # Shape of the AMPERE FAC data product
    assert sigmaP.shape==amp_sh, f"Shape of Pedersen conductance array is not (24,50). Got: {sigmaP.shape}"
    assert sigmaH.shape==amp_sh, f"Shape of Hall conductance array is not (24,50). Got: {sigmaH.shape}"
    mlat, mlt = np.array(np.abs(mlat), ndmin = 1), np.array(mlt, ndmin = 1)
    shape = np.broadcast(mlat, mlt).shape
    mlat, mlt = mlat.flatten(), mlt.flatten()
    
    #Arrays of the native AMPERE FAC product coordinates
    # amp_colat = np.tile(np.arange(1,amp_sh[1]+1),amp_sh[0]+1).reshape((amp_sh[0]+1,amp_sh[1]))
    # amp_mlt = np.array([amp_sh[1]*[i] for i in np.arange(0,amp_sh[0]+1)])
    sigmaH_ = np.vstack((sigmaH, sigmaH[0,:][np.newaxis,:]))
    sigmaP_ = np.vstack((sigmaP, sigmaP[0,:][np.newaxis,:]))

    
    #Spline representation of the global FAC pattern in the given hemisphere    
    hall_spline = RectBivariateSpline(np.arange(0,amp_sh[0]+1), 
                                        np.arange(1,amp_sh[1]+1), sigmaH_, 
                                        kx=1, ky=1)
    pedersen_spline = RectBivariateSpline(np.arange(0,amp_sh[0]+1), 
                                        np.arange(1,amp_sh[1]+1), sigmaP_, 
                                        kx=1, ky=1)
    
    #Evaluate spline representation at evaluation locations specified
    hall_interp = hall_spline.ev(mlt, 90-mlat)
    pedersen_interp = pedersen_spline.ev(mlt, 90-mlat)

    return (hall_interp.reshape(shape), pedersen_interp.reshape(shape))


def robinson_EUV(lon, lat, sigmaH, sigmaP, time, starlight = 0, F107 = 100,
              dipole=False, calibration = 'MoenBrekke1993'):
    """ calculate conductance at lat, lon for given time
    based on Robinson empirical FAC model + EUV contribution, from the functions
    defined above
    

    Parameters
    ----------
    lon: array
        geographic longitudes [deg]
    lat: array
        geograhpic latitudes [deg]
    sigmaH: 2D array
        Hall conductance [mho] on the native AMPERE grid.
    sigmaP: 2D array
        Pedersen conductance [mho] on the native AMPERE grid.          
    time: datetime
        time, used to get solar zenith angles and for apex coordinate conversion
    starlight: float, optional
        constant to add to conductance, often small (e.g. Strobel et al. 1980 https://doi.org/10.1016/0032-0633(80)90050-1)
        could be used to include "background-conductance" 
    F107: float, optional
        F107 index - used to scale EUV conductance. Default is 100
    dipole : bool, optional
        set to True if lat and lon are dipole coordinates. Default is False
    calibration: string, optional
        calibration to use in EUV_conductance calculation. See documentation
        of EUV_conductance function for info
        
    Returns
    -------
        Two arrays of conductances [mho] for each lat, lon, 
        one for total Hall and one for total Pedersen 
    
    """
    
    lat, lon = np.array(lat, ndmin = 1), np.array(lon, ndmin = 1)
    shape = np.broadcast(lat, lon).shape
    lat, lon = lat.flatten(), lon.flatten()
    
    cd = Dipole(time.year)       
    if dipole:
        mlat, mlon = lat, lon # input lat, lon is centered dipole
        lat, lon = cd.mag2geo(lat, lon) # to geographic
    else:
        a = apexpy.Apex(time, 110) 
        mlat, mlon = a.geo2apex(lat, lon, 110) # to mag
    mlt = cd.mlon2mlt(mlon, time)     # get mlt
    
    # solar zenith angles for EUV conductances
    sza = sunlight.sza(lat, lon, time)
        
    
    EUVh, EUVp = EUV_conductance(sza, F107, 'hp', calibration = calibration) # EUV
    rc_hall, rc_pedersen = robinson(mlat, mlt, sigmaH, sigmaP) # auroral
    
    return (np.sqrt(rc_hall**2 + EUVh**2 + starlight**2)).reshape(shape), (np.sqrt(rc_pedersen**2 + EUVp**2 + starlight**2)).reshape(shape)