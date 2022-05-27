"""
Conductance functions

"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from lompe.utils import sunlight
from apexpy import Apex

d2r = np.pi/180

def hardy_EUV(lon, lat, kp, time, hall_or_pedersen, F107 = 100, starlight = 0, calibration = 'MoenBrekke1993'):
    """ calculate conductance at lat, lon for given kp at given time
    based on Hardy model + EUV contribution, from the functions defined below

    parameters
    ----------
    lon: array
        geographic longitudes
    lat: array
        geograhpic latitudes
    kp: int
        Kp index (used for Hardy model)
    time: datetime
        time
    hall_or_pedersen: string
        'hall' or 'pedersen'
    F107: float, optional
        F107 index - used to scale EUV conductance. Default 100
    starlight: float, optional
        constant to add to conductance
    calibration: string, optional
        calibration to use in EUV_conductance calculation. See documentation
        of EUV_conductance function for info
    """
    if hall_or_pedersen.lower() not in ['hall', 'pedersen']:
        raise Exception('hardy_EUV: hall_or_pedersen must be either hall or pedersen')

    lat, lon = np.array(lat, ndmin = 1), np.array(lon, ndmin = 1)
    shape = np.broadcast(lat, lon).shape
    lat, lon = lat.flatten(), lon.flatten()

    sza = sunlight.sza(lat, lon, time)

    if hall_or_pedersen.lower() == 'hall':
        hop = 'h'
    if hall_or_pedersen.lower() == 'pedersen':
        hop = 'p'

    EUV = EUV_conductance(sza, F107, hop, calibration = calibration)

    a = Apex(time, refh = 110)
    mlat, mlon = a.geo2apex(lat, lon, 110)
    mlt = a.mlon2mlt(mlon, time)

    hc_hall, hc_pedersen = hardy(mlat, mlt, kp)

    if hall_or_pedersen.lower() == 'hall':
        return (hc_hall + EUV + starlight).reshape(shape)
    else:
        return (hc_pedersen + EUV + starlight).reshape(shape)



def EUV_conductance(sza, f107 = 100, hallOrPed = 'hp',
                    calibration = 'MoenBrekke1993'):
    """
    cond = EUV_conductance(sza, f107, hallOrPed, calibration='MoenBrekke1993')

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
        Solar zenith angle
    f107: float or array, optional
        F107 index - used to scale EUV conductance
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
        Hall conductances for each sza input value
    If hall_or_pedersen == 'p':
        Pedersen conductances for each sza input value
    If hall_or_pedersen == 'hp':
        Tuple of two arrays, one for Hall and one for Pedersen conductances, for each sza input value
    

    Example
    -------
    # Get Hall conductance
    f107 = 70
    sza = np.arange(0,120.1,0.1)
    hall = EUV_conductance(sza,f107,'h')

    # Get Pedersen conductance
    f107 = 70
    sza = np.arange(0,120.1,0.1)
    pedersen = EUV_conductance(sza,f107,'p')

    # Get Hall and Pedersen conductance
    f107 = 70
    sza = np.arange(0,120.1,0.1)
    hall, pedersen = EUV_conductance(sza,f107,'hp')


    References
    ----------
    Cousins, E. D. P., Matsuo, T. and Richmond, A. D. (2015) ‘Mapping high-latitude ionospheric 
    electrodynamics with SuperDARN and AMPERE’, Journal of Geophysical Research: Space Physics, 
    120, pp. 5854–5870. doi: 10.1002/2014JA020463.

    Moen, J. and Brekke, A. (1993) ‘The solar flux influence on quiet time conductances in the 
    auroral ionosphere’, Geophysical Research Letters, 20(10), pp. 971–974. doi: 10.1029/92GL02109.
    
    """
    shape = np.array(sza).shape

    assert hallOrPed.lower() in ['h','p','hp'],"Must select one of 'h', 'p', or 'hp' for hallOrPed!"


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
                                 f107**(f107hallexponent)*(0.81*PRODUCTION + 0.54*np.sqrt(PRODUCTION)),
                                 fill_value='extrapolate')
            sigh = halinterp(sza)

        if getP:
            pedinterp = interp1d(MODELSZAS,
                                 f107**(f107pedexponent)*(0.34*PRODUCTION + 0.93*np.sqrt(PRODUCTION)),
                                 fill_value='extrapolate')
            sigp = pedinterp(sza)

    else:

        if getH:
            halinterp = interp1d(MODELSZAS,
                                 f107**(f107hallexponent)*HalScl*(PRODUCTION)**(hallexponent),
                                 fill_value='extrapolate')
            sigh = halinterp(sza)

        if getP:
            pedinterp = interp1d(MODELSZAS,
                                 f107**(f107pedexponent)*PedScl*(PRODUCTION)**(pedexponent),
                                 fill_value='extrapolate')
            sigp = pedinterp(sza)


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


def hardy(mlat, mlt, kp):
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

    Returns
    -------
    Hall conductance: array
        array of Hall conductance [mho] with shape implied by mlat and mlt
    Peddersen conductance: array
        array of Pedersen conductance [mho] with shape implied by mlat and mlt

    """

    assert kp in [0, 1, 2, 3, 4, 5, 6]
    mlat, mlt = np.array(np.abs(mlat), ndmin = 1), np.array(mlt, ndmin = 1)
    shape = np.broadcast(mlat, mlt).shape
    mlat, mlt = mlat.flatten(), mlt.flatten()

    # load hall and pedersen coefficient files:
    basepath = os.path.dirname(__file__)

    hc = pd.read_table(basepath + '/../data/hardy_hall_coefficients.txt'    , sep = ',', skipinitialspace=True, skiprows = [0,])
    pc = pd.read_table(basepath + '/../data/hardy_pedersen_coefficients.txt', sep = ',', skipinitialspace=True, skiprows = [0,])

    # select only relevant kp:
    pc = pc[pc.Kp == 'K' + str(kp)]
    hc = hc[hc.Kp == 'K' + str(kp)]

    # define a dictionary whose keys are the 'term' column values, and values are the corresponding function of mlt
    pc['n']    = list(map(int, [t[-1] if t[-1] != 't' else 0 for t in pc['term']]))
    pc['trig'] = [np.sin if t[:3] == 'Sin' else np.cos  for t in pc['term']]    # the const term will be cos, but with n = 0
    hc['n']    = list(map(int, [t[-1] if t[-1] != 't' else 0 for t in hc['term']]))
    hc['trig'] = [np.sin if t[:3] == 'Sin' else np.cos  for t in hc['term']]

    # evaluate the fourier series:
    pedersen_epstein = dict(zip([u'maxvalue', u'maxlatitude', u'up-slope', u'down-slope'], [0]*4))
    for row in pc.iterrows():
        values = row[1]
        for key in pedersen_epstein:
            pedersen_epstein[key] += values[key] * values['trig'](values['n'] * mlt / 12 * np.pi)

    hall_epstein = dict(zip([u'maxvalue', u'maxlatitude', u'up-slope', u'down-slope'], [0]*4))
    for row in hc.iterrows():
        values = row[1]
        for key in hall_epstein:
            hall_epstein[key] += values[key] * values['trig'](values['n'] * mlt / 12 * np.pi)

    # evaluate the Epstein transition function, Pedersen:
    r, S1, S2, h0 = pedersen_epstein['maxvalue'], pedersen_epstein['up-slope'], pedersen_epstein['down-slope'], pedersen_epstein['maxlatitude']
    pedersen_conductance = r + S1*(mlat - h0) + (S2 - S1) * np.log((1 - S1/(S2 * np.exp(-(mlat - h0)))) / (1 - (S1/S2)))

    # evaluate the Epstein transition function, Halle:
    r, S1, S2, h0 = hall_epstein['maxvalue'], hall_epstein['up-slope'], hall_epstein['down-slope'], hall_epstein['maxlatitude']
    hall_conductance = r + S1*(mlat - h0) + (S2 - S1) * np.log((1 - S1/(S2 * np.exp(-(mlat - h0)))) / (1 - (S1/S2)))

    # introduce floors (using recommendation from paper)
    pedersen_conductance[(mlat < h0) & (pedersen_conductance < 0   )] = 0
    pedersen_conductance[(mlat > h0) & (pedersen_conductance < 0.55)] = 0.55
    hall_conductance[    (mlat < h0) & (hall_conductance     < 0   )] = 0
    hall_conductance[    (mlat > h0) & (hall_conductance     < 0.55)] = 0.55

    return hall_conductance.reshape(shape), pedersen_conductance.reshape(shape)

