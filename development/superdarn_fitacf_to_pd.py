#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Read 2hr zipped FITACF-file into lompe-format

Created on Mon Jun 28 13:17:41 2021

@author: amalie
"""

import numpy as np
import pydarnio
import pydarn
from pydarn import build_scan, time2datetime, radar_fov
import pandas as pd
import glob
import bz2
import datetime as dt

############CHANGE THESE:
time = dt.datetime(2014,12,15,0)
rads = ['rkn', 'cly', 'inv']
sdpath = ''   # path to folder containing fitacf-files
savepath = '' # path for output files
##########

def load_file(stime, rad, sdpath):
    """
    Reads FITACF 2.5 file from local path

    Parameters
    ----------
    stime : datetime
        time within file
    rad : str
        radar abbr., e.g., 'rkn'
    sdpath : str
        path to bz2 files

    Returns
    -------
    fitacf_data : list[dict]
        2hrs of superDARN data

    """

    hr = int(f'{stime:%H}') #file starts with even hour, NB: last entry is first entry of hour
    if hr % 2 != 0:
        stime = stime - dt.timedelta(minutes=60)

    #Example fitacf filename: 20141201.0001.00.cve.fitacf.bz2
    pattern = f'{stime:%Y%m%d.%H}' + '*' + rad
    files = glob.glob(sdpath + '/' + rad + '/' + pattern + '*' + 'fitacf.bz2')
    fitacf_file = files[0]

    with bz2.open(fitacf_file) as fp:
        fitacf_stream = fp.read()
    sdarn_read = pydarnio.SDarnRead(fitacf_stream, True)
    fitacf_data = sdarn_read.read_fitacf()      #reads fitacf_file into list[dict]

    return fitacf_data

def vec_getbearing(lat0, lon0, lat1, lon1):
    """
    Calculate the starting bearing angle along great circle from lat0,lon0 to lat1,lon1
    NB: input must be degrees

    Parameters
    ----------
    lat0 : array
        start lats, degrees (either geographic or magnetic)
    lon0 : array
        start lons, degrees (either geographic or magnetic)
    lat1 : array
        end lats, degrees (either geographic or magnetic)
    lon1 : array
        end lons, degrees (either geographic or magnetic)

    Returns
    -------
    bearing : array
        azimuth angles in radians

    """

    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)

    coslt1 = np.cos(lat1)
    sinlt1 = np.sin(lat1)
    coslt0 = np.cos(lat0)
    sinlt0 = np.sin(lat0)
    cosl0l1 = np.cos(lon1-lon0)
    sinl0l1 = np.sin(lon1-lon0)

    cosc = (sinlt0 * sinlt1) + (coslt0 * coslt1 * cosl0l1)

    # Avoid roundoff problems by clamping cosine range to [-1,1].
    negs = cosc < -1.
    cosc[negs] = -1.
    poss = cosc > 1.
    cosc[poss] = 1.

    sinc = np.sqrt(1.0 - cosc**2)

    cosaz = np.zeros(len(lat0))     #cosine azimuth
    sinaz = np.zeros(len(lat0))     #sine azimuth

    #for "large" angles
    large = np.abs(sinc) > 1.0e-7
    cosaz[large] = ((coslt0[large] * sinlt1[large]) - (sinlt0[large]*coslt1[large]*cosl0l1[large])) / sinc[large]
    sinaz[large] = (sinl0l1[large]*coslt1[large])/sinc[large]

    #"small" angle approximation.
    small = np.abs(sinc) <= 1.0e-7
    cosaz[small] = 1.0
    sinaz[small] = 0.0

    return np.arctan2(sinaz, cosaz)     #azimuth in radians

def getscan(time, fitacf_data):
    """
    Gets single scan of SuperDARN data and calculates azimuth angle (geographic)

    Parameters
    ----------
    time : datetime
        time of scan
    fitacf_data : list[dict]
        2hrs of SuperDARN FITACF 2.5 data from load_file

    Returns
    -------
    vdict : dict
        Contains zeros if radar scan is missing

    """
    radars = pydarn.utils.superdarn_radars.SuperDARNRadars()    # radar info

    scan_index = time

    stid = fitacf_data[0]['stid']   # station code
    beam_corners_geo_lats, beam_corners_geo_lons = radar_fov(stid, coords = 'geo')  # FOV grid
    fan_shape = beam_corners_geo_lons.shape

    beam_scan = build_scan(fitacf_data)
    if isinstance(scan_index, dt.datetime): # list where scans start
        # loop through dmap_data records, dump a datetime
        scan_time = scan_index
        scan_index = 0
        found_match = False
        for rec in fitacf_data:
            rec_time  = time2datetime(rec)
            if abs(rec['scan']) == 1:
                scan_index += 1
            # Need the abs since you cannot have negative seconds
            diff_time = abs(scan_time - rec_time)
            if diff_time.seconds < 1:
                found_match = True
                break
        # handle datetimes out of bounds
        if found_match == False:
            fillval = np.zeros((fan_shape[0] - 1, fan_shape[1] - 1))
            vdict = {'vel' : fillval , 'err' : fillval, 'ground' : fillval, 'lats' : fillval,\
                     'lon' : fillval, 'azimuth' : fillval, 'time' : time, 'freqs' : 0, 'stid' : stid}

    plot_beams = np.where(beam_scan == scan_index)  # scan to get

    sdtime = dt.datetime(fitacf_data[plot_beams[0][0]]['time.yr'],
                            fitacf_data[plot_beams[0][0]]['time.mo'],
                            fitacf_data[plot_beams[0][0]]['time.dy'],
                            fitacf_data[plot_beams[0][0]]['time.hr'],
                            fitacf_data[plot_beams[0][0]]['time.mt'],
                            fitacf_data[plot_beams[0][0]]['time.sc'])
    freq = fitacf_data[0]['tfreq']


    beam_lats = np.zeros((fan_shape[0]-1, fan_shape[1]-1))
    beam_lons = np.zeros((fan_shape[0]-1, fan_shape[1]-1))
    #TODO: fix issue with meridians
    for beam in range(fan_shape[1]-1):  # get center of FOV grid cells
        for gate in range(fan_shape[0]-1):
            beam_lats[gate, beam] = np.mean([beam_corners_geo_lats[gate, beam], \
                            beam_corners_geo_lats[gate+1, beam], beam_corners_geo_lats[gate, beam+1], \
                            beam_corners_geo_lats[gate+1, beam+1]])
            beam_lons[gate, beam] = np.mean([beam_corners_geo_lons[gate, beam], \
                            beam_corners_geo_lons[gate+1, beam], beam_corners_geo_lons[gate, beam+1], \
                            beam_corners_geo_lons[gate+1, beam+1]])

    #Jone changed from [4][2][:], Amalie changed back
    #TODO: different versions of the pydarn libraries?
    sitelat, sitelon, sitealt = radars.radars[stid][4][2][:]
    # [3][2][0], radars.radars[stid][3][2][1], radars.radars[stid][3][2][2]  # boresitre info for azimuth calculation
    sitelats = np.ones(beam_lats.shape)*sitelat
    sitelons = np.ones(beam_lats.shape)*sitelon

    scan = np.zeros((fan_shape[0] - 1, fan_shape[1]-1))     # scan ('v'-velocity) [m/s]
    v_err = np.zeros((fan_shape[0] - 1, fan_shape[1]-1))    # error in velocity [m/s]
    grndsct = np.zeros((fan_shape[0] - 1, fan_shape[1]-1))  # ground backscatter flags
    azimuth = np.zeros((fan_shape[0] - 1, fan_shape[1]-1))  # geographic azimuth angle [deg]
    for i in np.nditer(plot_beams):
            try:
                slist = fitacf_data[i.astype(int)]['slist'] # get a list of gates where there is data
                beam = fitacf_data[i.astype(int)]['bmnum']  # get the beam number for the record
                scan[slist, beam] = fitacf_data[i.astype(int)]['v']
                v_err[slist, beam] = fitacf_data[i.astype(int)]['v_e']
                grndsct[slist, beam] = fitacf_data[i.astype(int)]['gflg']
                azimuth[slist, beam] = np.degrees(vec_getbearing(beam_lats[slist, beam],\
                                                                 beam_lons[slist, beam],\
                                                                     sitelats[slist, beam], \
                                                                         sitelons[slist, beam]))
                for k in slist:     # azimuth corrections
                    if scan[k, beam] < 0:
                        azimuth[k, beam] = azimuth[k, beam] + 180.
                        scan[k, beam] = np.abs(scan[k, beam]) #2021-10-21: jreistad added this to be consistent with the grid-files
                    if azimuth[k, beam] > 180:
                        azimuth[k, beam] = azimuth[k, beam] - 360.
            except:
                continue             # if there is no slist field this means partial record

    # dict of single scan
    vdict = {'vel' : scan, 'err' : v_err, 'ground' : grndsct, 'lat' : beam_lats, 'lon' : beam_lons, \
             'azimuth' : azimuth, 'time' : sdtime, 'freqs' : freq, 'stid' : stid}

    return vdict


ddd = pd.DataFrame()
for rad in rads:    # one 2hr file at the time
    data = load_file(time, rad, sdpath)

    stime = dt.datetime(data[0]['time.yr'],
                        data[0]['time.mo'],
                        data[0]['time.dy'],
                        data[0]['time.hr'],
                        data[0]['time.mt'],
                        data[0]['time.sc'])

    etime = dt.datetime(data[-1]['time.yr'],
                        data[-1]['time.mo'],
                        data[-1]['time.dy'],
                        data[-1]['time.hr'],
                        data[-1]['time.mt'],
                        data[-1]['time.sc'])
    duration = etime - stime
    dur = duration.seconds//60

    stimes = []
    for i in range(0,dur):
        stimes.append(stime + dt.timedelta(minutes=i))

    for scantime in stimes:
        vdict = getscan(scantime, data)
        use = (vdict['vel'] != 0) & (vdict['ground'] != 1)  # removes v=0 m/s, and ground backscatter
        temp = pd.DataFrame()
        stid = [vdict['stid']]*(vdict['azimuth'][use].flatten()).shape[0]
        temp.loc[:,'stid'] = stid
        temp.loc[:,'vlos'] = vdict['vel'][use].flatten()
        temp.loc[:,'error'] = vdict['err'][use].flatten()
        temp.loc[:,'glat'] = vdict['lat'][use].flatten()
        temp.loc[:,'glon'] = vdict['lon'][use].flatten()
        temp.loc[:,'azimuth'] = vdict['azimuth'][use].flatten()
        timelist = [vdict['time']]*(vdict['azimuth'][use].flatten()).shape[0]
        temp.loc[:,'time'] = timelist
        ddd = ddd.append(temp)

ddd.index = ddd.time
ddd['le'], ddd['ln'] = np.sin(ddd['azimuth'] * np.pi / 180), np.cos(ddd['azimuth'] * np.pi / 180)

pd.to_pickle(ddd, savepath + f'{time:%Y%m%d-%H}' + '-superDARN.pd', protocol=4)
