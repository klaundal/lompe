""" Functions that gather data from input time interval and return dataset that
is to be used with lompe. 

    - read_ssusi   : DMSP sat auroral images       (auroral conductance)
    - read_ssies   : DMSP sat plasma convection    (LOS convection)
    - read_sdarn   : SuperDARN plasma convection   (LOS convection)
    - read_smag    : SuperMAG magnetic field       (groundmag)
    - read_iridium : Iridium sat magnetic field    (spacemag)

Helpers
----------
    - getbearing
    - cross_track_los
    - radar_losvec_from_mag
    - los_azimuth2en
    
"""

import glob
import os
import pandas as pd
import datetime as dt
import xarray as xr
import numpy as np
from lompe.utils.time import date2doy
# degrees <-> radians conversion
d2r = np.pi / 180.
r2d = 180. / np.pi


def read_ssusi(event, hemi='north', basepath='./', tempfile_path='./', source='jhuapl'):
    """

    Called and used for modelling auroral conductance in cmodel.py.

    Extract the relevant SSUSI info from netcdf-files downloaded from APL server.
    https://ssusi.jhuapl.edu/data_products (EDR AURORA)
    Will load all SSUSI data from specified hemisphere into xarray object. 
    Any DMSP Block 5D satellite (F16-19) available in folder will be used.

    Parameters
    ----------
    event : str
        string on format 'yyyy-mm-dd' to specify time of event for model.
    hemi : str, optional
        specify hemisphere 
        Default: 'north'
    basepath : str, optional
        location of netcdf-files downloaded from APL server (.NC).
        Default: './'
    tempfile_path : str, optional
        Location for storing processed SSUSI data.
        Default: './'
    source : str, optional
        Specify source of SUSSI data. 
        Default: 'jhuapl', other option is 'cdaweb'

    Returns
    -------
    savefile : str
        Path to saved file containing SSUSI data + conductances extracted from the images.

    """
    '''
    Extract the relevant info from netcdf-files downloaded from APL server:
    Note that these data are gridded on a 363,363 geomagnetic grid. We assume this is AACGM,
    based on email correspondence with L. Paxton. This is not clear from the documentation.
    No geographic information of these gridded data exist in the EDR aurora files downloaded
    Quote from documentation: "This array is a uniform grid in a polar azimuthal equidistant
    projection of the geomagnetic latitude and geomagnetic local time, with appropriate
    geomagnetic pole as the origin; that is: 90-ABS(glat) is the radius and geomagnetic
    local time is the angle of a two-dimensional polar coordinate system."
    Hence, the mapped latitude array (363x363) is the same for both hemispheres, and
    the values are therefore always positive.
    '''
    if not basepath.endswith('/'):
        basepath += '/'
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'

    # check if the processed file exists
    savefile = tempfile_path + \
        event.replace('-', '') + '_ssusi_' + hemi + '.nc'
    if os.path.isfile(savefile):
        return savefile

    try:
        import netCDF4
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'read_ssusi: Could not load netCDF4 module. Will not be able to read SSUSI-files from APL.')

    imgs = []  # List to hold all images. Converted to xarray later.
    doy_str = f"{date2doy(event):03d}"

    for sat in ['F16', 'F17', 'F18', 'F19']:
        if source == 'jhuapl':
            files = glob.glob(basepath + '*' + sat + '*' +
                              event[0:4] + event[5:7] + event[8:10] + '*.NC')
        elif source == 'cdaweb':
            files = glob.glob(basepath + '*' + sat + '*' +
                              event[0:4] + doy_str + '*.nc')
        files.sort()
        if len(files) == 0:
            continue

        ii = 0  # counter
        ii_max = len(files)
        ii_max_orbit = files[-1].split('_SN.')[-1].split('-')[0]

        extra_orbit = format(int(ii_max_orbit) + 1, '05')
        extra_file = glob.glob(basepath + '*' + sat +
                               '*_SN.' + extra_orbit + '-' + '*.NC')

        if len(extra_file) == 1:
            files.append(extra_file[0])

        for file in files:
            f = netCDF4.Dataset(file)

            mlat = f.variables['LATITUDE_GEOMAGNETIC_GRID_MAP'][:]
            mlon = f.variables['LONGITUDE_GEOMAGNETIC_' +
                               hemi.upper() + '_GRID_MAP'][:]
            mlt = f.variables['MLT_GRID_MAP'][:]
            uthr = f.variables['UT_' + hemi.upper()[0]][:]
            doy = int(f.variables['DOY'][:])
            year = int(f.variables['YEAR'][:])

            wavelengths = f.variables['DISK_RADIANCEDATA_INTENSITY_' + hemi.upper()][:]
            char_energy = f.variables['ELECTRON_MEAN_' +
                                      hemi.upper() + '_ENERGY_MAP'][:]
            energyflux = f.variables['ENERGY_FLUX_' + hemi.upper() + '_MAP'][:]

            f.close()

            mask = uthr == 0

            uthr[mask] = np.nan
            wavelengths[:, mask] = np.nan
            char_energy[mask] = np.nan
            energyflux[mask] = np.nan
            mlon[mask] = np.nan

            # Applying Robinson formulas to calculate conductances: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JA092iA03p02565
            SP = (40. * char_energy * np.sqrt(energyflux)) / \
                (16. + char_energy**2)
            SH = 0.45 * char_energy**0.85 * SP

            if sum(np.isfinite(uthr[:, 181]).flatten()) > 0:  # there is data in north
                # Calculate center time:
                # there is a sudden transition across midnight in the pass
                if np.nanstd(uthr) > 0.5:
                    next_day = (uthr > 0) & (uthr < 2)
                    if sum(next_day.flatten()) > 0:
                        uthr[next_day] = uthr[next_day] + 24.
                center_hr = np.nanmean(uthr, axis=0)[181]
                if (ii == 0) & (center_hr >= 22):
                    ii = + 1
                    # the image is (likely) from the previous day
                    continue

                if (ii == ii_max) & (center_hr < 22):
                    ii = + 1
                    continue              # the image is not from the same day and is skipped

                if (ii == ii_max):
                    doy = doy - 1

                hr = int(center_hr)
                if hr >= 24:
                    hr = hr - 24
                    center_hr = center_hr - 24

                m = int((center_hr - hr) * 60)
                s = round(((center_hr - hr) * 60 - m) * 60)
                if s == 60:
                    t0 = dt.datetime(year, 1, 1, hr, m, 59) + \
                        dt.timedelta(seconds=1)
                else:
                    t0 = dt.datetime(year, 1, 1, hr, m, s)

                dtime = t0 + dt.timedelta(doy - 1)
                # put into xarray object
                img = xr.Dataset({'uthr': (['row', 'col'], uthr),
                                  'mlon': (['row', 'col'], mlon),
                                  'mlat': (['row', 'col'], mlat),
                                  'mlt': (['row', 'col'], mlt),
                                  '1216': (['row', 'col'], wavelengths[0, :, :]),
                                  '1304': (['row', 'col'], wavelengths[1, :, :]),
                                  '1356': (['row', 'col'], wavelengths[2, :, :]),
                                  'lbhs': (['row', 'col'], wavelengths[3, :, :]),
                                  'lbhl': (['row', 'col'], wavelengths[4, :, :]),
                                  'E0': (['row', 'col'], char_energy),
                                  'je': (['row', 'col'], energyflux),
                                  'SP': (['row', 'col'], SP),
                                  'SH': (['row', 'col'], SH)})
                img = img.expand_dims(date=[dtime])
                img = img.assign({'satellite': sat})
                img = img.assign(
                    {'orbit': file.split('_SN.')[-1].split('-')[0]})
                imgs.append(img)
            ii += 1

    if len(imgs) == 0:
        print('No SSUSI images found.')
        return None

    else:     # save as netcdf in specified path
        imgs = xr.concat(imgs, dim='date')
        imgs = imgs.assign({'hemisphere': hemi})
        imgs = imgs.sortby(imgs['date'])
        print('DMSP SSUSI file saved: ' + savefile)

        imgs.to_netcdf(savefile)

        return savefile


def read_ssies(event, sat, basepath='./', tempfile_path='./', forcenew=False, **madrigal_kwargs):
    """
    Download DMSP SSIES ion drift meter data for a full day and get relevant parameters for Lompe. 
    Saves hdf file in tempfile_path and returns path

    Note that Madrigal needs user specificatyions through **madrigal_kwargs. 
    There is no need to create user beforehand.
    E.g. {'user_fullname' : 'First Last', 'user_email' : 'name@host.com', 'user_affiliation' : 'University'}

    Example usage for Lompe:
    fn = dataloader.read_ssies(event, 17, tempfile_path=tempfile_path, **madrigal_kwargs)
    ssies = pd.read_hdf(fn, mode='r')
    ssies = ssies[(ssies.quality==1) | (ssies.quality==2)]
    ssies = ssies[stime - 2*DT : stime + 2*DT].dropna()
    v_crosstrack = np.abs(ssies.hor_ion_v).values
    coords = np.vstack((ssies.glon.values, ssies.gdlat.values))
    los  = np.vstack((ssies['le'].values, ssies['ln'].values))
    ssies_data = lompe.Data(v_crosstrack, coords, datatype = 'convection', scale = 500, LOS=los)

    Parameters
    ----------
    event : str
        string on format 'yyyy-mm-dd' to specify time of event for model.
    sat : str
        Satellite ID for DMSP Block 5D satellite (16-19)
    basepath : str, optional
        path to raw files. currently is only for temporary storage of files from madrigal
        Default: './'
    tempfile_path : str, optional
       Path to dir where processed hdf files are placed
       Default: './'
    forcenew : bool, optional
        Force the function to download the data even if file exists
        Default: False.
    **madrigalkwargs : dict
        needed to download data from MadrigalWeb
        Example: {'user_fullname' : 'First Last', 'user_email' : 'name@host.com', 'user_affiliation' : 'University'}

    Returns
    -------
    savefile : str
        Path to hdf-file containing SSIES data for Lompe.

    """
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'
    if not basepath.endswith('/'):
        basepath += '/'

    savefile = tempfile_path + \
        event.replace('-', '') + '_ssies_f' + str(sat) + '.h5'

    # do not download if file already exists
    if os.path.isfile(savefile) and not forcenew:
        return savefile

    # imports
    import calendar

    try:
        from scipy import interpolate
    except ModuleNotFoundError:
        raise ModuleNotFoundError('read_ssies requires scipy module.')

        # silence NaturalNameWarnings
    import warnings
    from tables import NaturalNameWarning
    warnings.filterwarnings('ignore', category=NaturalNameWarning)

    # If file does not exist already, we need API to download
    try:
        # API, http://cedar.openmadrigal.org/docs/name/rt_contents.html
        import madrigalWeb.madrigalWeb
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'read_ssies: Could not import MadrigalWeb module. Will not be able to download DMSP SSIES files.')

    madrigalUrl = 'http://cedar.openmadrigal.org'
    # madrigalUrl = 'http://madrigal.haystack.mit.edu/madrigal'
    try:
        testData = madrigalWeb.madrigalWeb.MadrigalData(madrigalUrl)
    except:
        raise RuntimeError(
            'Madrigal site is not working. Try a manual download.')

    # specify one day download
    sTime = dt.datetime(int(event[0:4]), int(
        event[5:7]), int(event[8:10]), 0, 0)
    eTime = sTime + dt.timedelta(days=1)
    usTime = calendar.timegm(sTime.utctimetuple())
    ueTime = calendar.timegm(eTime.utctimetuple())

    expList = testData.getExperiments(8100, sTime.year, sTime.month, sTime.day, sTime.hour, sTime.minute, 0,
                                      eTime.year, eTime.month, eTime.day, eTime.hour, eTime.minute, 0)

    dmsp = pd.DataFrame()
    dmsp2 = pd.DataFrame()   # for the density fraction
    date_str = event.replace('-', '')
    no_data_found = True
    for i in expList:
        fileList = testData.getExperimentFiles(i.id)
        filenames = []

        for fname in fileList:
            filenames.append(fname.name)
        filenames = [filename for filename in filenames if date_str in filename]
        if len(filenames) == 0:
            continue
        no_data_found = False
            # ssies = str([s for s in filenames if '_' + str(sat) + 's1' in s][0])
        ssies = [s for s in filenames if '_' + str(sat) +  's1.' in s]
        # temp_dens = str(
        #     [s for s in filenames if '_' + str(sat) + 's4.' in s][0])
        temp_dens = [s for s in filenames if '_' + str(sat) + 's4.' in s]

        datafile = basepath + 'ssies_temp_' + event + '.hdf5'
        result = testData.downloadFile(
            ssies[0], datafile, **madrigal_kwargs, format="hdf5")
        f = pd.read_hdf(datafile, mode='r', key='Data/Table Layout')

        tempdensfile = basepath + 'ssies_tempdens_data_' + event + '.hdf5'
        result = testData.downloadFile(
            temp_dens[0], tempdensfile, **madrigal_kwargs, format="hdf5")
        f2 = pd.read_hdf(tempdensfile, mode='r', key='Data/Table Layout')

        use = (f.ut1_unix >= usTime) & (f.ut1_unix < ueTime)
        use2 = (f2.ut1_unix >= usTime) & (f2.ut1_unix < ueTime)
        temp = f[use]       # pd.DataFrame()
        temp2 = f2[use2]    # pd.DataFrame()

        dmsp = pd.concat([dmsp, temp])
        dmsp2 = pd.concat([dmsp2, temp2])

        dmsp.index = np.arange(len(dmsp))
        dmsp2.index = np.arange(len(dmsp2))
    if no_data_found:
        print('No data found for ' + event + ' and satellite ' + str(sat))
        return None

    # set datetime as index
    times = []
    for i in range(len(dmsp)):
        times.append(dt.datetime.utcfromtimestamp(int(dmsp['ut1_unix'][i])))
    dmsp.index = times

    # reindexing due to lower cadence measurements
    times2 = []
    for i in range(len(dmsp2)):
        times2.append(dt.datetime.utcfromtimestamp(int(dmsp2['ut1_unix'][i])))
    dmsp2.index = times2
    dmsp2 = dmsp2.reindex(index=dmsp.index, method='nearest', tolerance='2sec')

    dmsp.loc[:, 'po+'] = dmsp2['po+']
    dmsp.loc[:, 'te'] = dmsp2['te']
    dmsp.loc[:, 'ti'] = dmsp2['ti']

    # Smooth the orbit
    import matplotlib

    # latitude (geodetic)
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.gdlat[::60], (dmsp.gdlat[-1:])]).values, k=2)
    gdlat = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # longitude (geodetic)
    negs = dmsp.glon < 0
    dmsp.loc[negs, 'glon'] = dmsp.glon[negs] + 360
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.glon[::60], (dmsp.glon[-1:])]).values, k=2)
    glon = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # altitude (geodetic)
    tck = interpolate.splrep(matplotlib.dates.date2num(dmsp.index[::60].append(dmsp.index[-1:])),
                             pd.concat([dmsp.gdalt[::60], (dmsp.gdalt[-1:])]).values, k=2)
    gdalt = interpolate.splev(matplotlib.dates.date2num(dmsp.index), tck)

    # get eastward and northward component of cross track direction
    le, ln, bearing = cross_track_los(dmsp['hor_ion_v'], gdlat, glon)

    # put together the relevant data
    ddd = pd.DataFrame()      # dataframe to return
    ddd.index = dmsp.index
    ddd.loc[:, 'gdlat'] = gdlat
    ddd.loc[:, 'glon'] = glon
    ddd.loc[:, 'gdalt'] = gdalt
    ddd.loc[:, 'hor_ion_v'] = np.abs(dmsp.hor_ion_v)
    ddd.loc[:, 'vert_ion_v'] = dmsp.vert_ion_v
    ddd.loc[:, 'bearing'] = bearing
    ddd.loc[:, 'le'] = le
    ddd.loc[:, 'ln'] = ln

    # quality flag from Zhu et al 2020 page 8 https://doi.org/10.1029/2019JA027270
    # flag1 is good, flag2 is also usually acceptable

    flag1 = (dmsp['ne'] > 1e9) & (dmsp['po+'] > 0.85)
    flag2 = ((dmsp['po+'] > 0.85) & (dmsp['ne'] > 1e8) & (dmsp['ne'] < 1e9)) | ((dmsp['po+'] > 0.75)
                                                                                & (dmsp['po+'] < 0.85) & (dmsp['ne'] > 1e8))
    flag3 = (dmsp['ne'] < 1e8) | (dmsp['po+'] < 0.75)
    flag4 = dmsp['po+'].isna()

    ddd.loc[:, 'quality'] = np.zeros(ddd.shape[0]) + np.nan
    ddd.loc[flag1, 'quality'] = 1
    ddd.loc[flag2, 'quality'] = 2
    ddd.loc[flag3, 'quality'] = 3
    ddd.loc[flag4, 'quality'] = 4

    ddd.to_hdf(savefile, key='df', mode='w')
    print('DMSP SSIES file saved: ' + savefile)

    # remove temporary files
    os.remove(datafile)
    os.remove(tempdensfile)

    return savefile


def read_sdarn(event, basepath='./', tempfile_path='./', hemi='north'):
    """
    Will load all data from specified day into pandas dataframe. 
    Currently, we use the gridded and cleaned dataset from E. Thomas (downloaded from https://zenodo.org/record/3618607#.YD4KiXVKiEJ )
    Saves hdf file in tempfile_path and returns path

    Parameters
    ----------
    event : str
        string on format 'yyyy-mm-dd' to specify time of event for model.
    basepath : str, optional
        location of downloaded SuperDARN gridmap files
        Default: './'
    tempfile_path : str, optional
        path to already processed SuperDARN files (hdf)
        Default: './'
    hemi : str, optional
        specify hemisphere 
        Default: 'north'

    Returns
    -------
    savefile : str
        Path to hdf-file containing SuperDARN data for Lompe.

    """

    assert len(basepath) != 0, 'the path to the SuperDARN file cannot be empty'

    if len(tempfile_path) == 0:
        tempfile_path = './'

    if not basepath.endswith('/'):
        basepath += '/'
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'

    # check if the processed file exists
    savefile = tempfile_path + event.replace('-', '') + '_superdarn_grdmap.h5'
    if os.path.isfile(savefile):
        return savefile

    # local imports
    try:
        import apexpy
    except ModuleNotFoundError:
        raise ModuleNotFoundError('read_sdarn requires apexpy module.')

    try:
        import pydarn  # https://zenodo.org/record/3727269
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'read_sdarn: Could not import pydarn module. Will not be able to read SuperDARN-files.')

    # open the file
    file = glob.glob(basepath + event.replace('-', '') + '.' + hemi[0] + '*')
    if len(file) == 0:
        print('SuperDARN data not found in basepath.')
        return None

    else:
        file = file[0]

    SDarn_read = pydarn.SuperDARNRead(file)
    if file.split('.')[-1] == 'grdmap':
        data = SDarn_read.read_grid()
    elif file.split('.')[-1] == 'map':
        data = SDarn_read.read_map()
    else:
        print('SuperDARN file has unknown format. Aborting.')
        return None

    # radar information (site coordniates)
    radars = pydarn.SuperDARNRadars.radars

    ddd = pd.DataFrame()
    for t in range(len(data)):
        tt = data[t]
        if sum(tt['nvec']) == 0:
            continue

        stime = dt.datetime(tt['start.year'], tt['start.month'], tt['start.day'],
                            tt['start.hour'], tt['start.minute'], int(tt['start.second']))
        etime = dt.datetime(tt['end.year'], tt['end.month'], tt['end.day'],
                            tt['end.hour'], tt['end.minute'], int(tt['end.second']))
        duration = etime - stime

        temp = pd.DataFrame()
        for c, s in enumerate(tt['stid']):
            ne = tt['nvec'][c]
            stids = (np.ones(ne) * s).astype(int)
            freqs = np.ones(ne) * tt['freq'][c]
            names = [radars[s][0]] * ne

            temp2 = pd.DataFrame()
            temp2.loc[:, 'stid'] = stids
            temp2.loc[:, 'radar'] = names
            temp2.loc[:, 'freq'] = freqs

            if ne > 0:
                temp2.loc[:, 'stime'] = stime
                temp2.loc[:, 'etime'] = etime
                temp2.loc[:, 'duration'] = duration
                temp = pd.concat([temp, temp2], ignore_index=True)

        temp.loc[:, 'mlat'] = tt['vector.mlat']              # in degrees AACGM
        temp.loc[:, 'mlon'] = tt['vector.mlon']              # in degrees AACGM
        # in degrees, the angle between los and magnetic north
        temp.loc[:, 'azimuth'] = tt['vector.kvect']
        temp.loc[:, 'vlos'] = tt['vector.vel.median']        # in m/s
        temp.loc[:, 'vlos_sd'] = tt['vector.vel.sd']         # in m/s
        temp.loc[:, 'range'] = tt['vector.pwr.median']       # in km
        # spectral width in m/s
        temp.loc[:, 'wdt'] = tt['vector.wdt.median']
        temp.loc[:, 'time'] = stime + duration / 2

        ddd = pd.concat([ddd, temp], ignore_index=True)

    ddd.index = ddd.time

    # get line-of-sight unit vector in geographic coords
    ddd['glat'], ddd['glon'], ddd['le'], ddd['ln'], ddd['le_m'], ddd['ln_m'] = radar_losvec_from_mag(ddd['mlat'].values,
                                                                                                     ddd['mlon'].values, ddd['azimuth'].values, ddd.index[0])

    # save dataframe for later use
    ddd.to_hdf(savefile, key='df', mode='w')
    print('SuperDARN file saved: ' + savefile)

    return savefile


def read_smag(event, basepath='./', tempfile_path='./', file_name=''):
    """
    Gets relevant ground magnetometer data from specified local SuperMAG file: https://supermag.jhuapl.edu/
    (baseline removed)

    Filename does not contain date info when downloading netcdf from SuperMAG website manually. User 
    needs to specify filename.

    Parameters
    ----------
    event : str
        string on format 'yyyy-mm-dd' to specify time of event for model.
    basepath : str, optional
        location of downloaded SuperMAG files (netcdf)
        Default: './'
    tempfile_path : str, optional
        path to already processed SuperMAG files (hdf)
        Default: './'
    file_name : str, optional
        Name of existing file.
        Default: ''

    Returns
    -------
    savefile : str
        Path to hdf-file containing SuperMAG data for Lompe.

    """

    if not basepath.endswith('/'):
        basepath += '/'
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'

    # check if the processed file exists
    savefile = tempfile_path + event.replace('-', '') + '_supermag.h5'
    if os.path.isfile(savefile):
        return savefile

    # find netcdf in basepath
    if not os.path.isfile(basepath + file_name):
        file_name = event.replace('-', '') + '_supermag.netcdf4'
    if not os.path.isfile(basepath + file_name):
        file_name = event.replace('-', '') + '_supermag.netcdf'
    if not os.path.isfile(basepath + file_name):
        raise FileNotFoundError(
            'Cannot find SuperMAG netcdf in specified folder. Please specify the correct path and file name.')

    # read netcdf and make hdf for Lompe
    elif file_name.endswith('.netcdf4') | file_name.endswith('.netcdf'):

        # From downloaded netcdf file:
        sm = xr.load_dataset(basepath + file_name,
                             decode_coords=False, engine='netcdf4')

        # convert time columns to datetimes:
        times = [dt.datetime(*x) for x in zip(sm.time_yr.values, sm.time_mo.values,
                                              sm.time_dy.values, sm.time_hr.values, sm.time_mt.values)]
        times = np.array(pd.DatetimeIndex(times)).reshape((-1, 1))
        times = np.repeat(times, sm.vector.size, axis=1).flatten()

        df = pd.DataFrame({'Be': sm.dbe_geo.values.flatten(), 'Bn': sm.dbn_geo.values.flatten(), 'Bu': -sm.dbz_geo.values.flatten(),
                           'lat': sm.glat.values.flatten(), 'lon': sm.glon.values.flatten()},
                          index=times).sort_index()
        df = df.dropna()

        # save the dataset
        finishedfile = tempfile_path + event.replace('-', '') + '_supermag.h5'
        df.to_hdf(finishedfile, key='df', mode='w')
        print('SuperMAG file saved: ' + finishedfile)

        return finishedfile

    else:
        raise ValueError('Something went wrong.')


def read_iridium(event, basepath='./', tempfile_path='./', file_name=''):
    """
    Convert netcdf files (dB raw) downladed from the AMPERE website (http://ampere.jhuapl.edu/dataraw/index.html) 
    to format used by Lompe.

    Parameters
    ----------
    event : str
        string on format 'yyyy-mm-dd' to specify time of event for model.
    basepath : str, optional
        location of netcdf-files downloaded from AMPERE website - fitted data (.ncdf).
        Default: './'
    tempfile_path : str, optional
        location for saving processed data files for event date. Will look here for
        already processed data sets
        Default: './'
    file_name : str, optional
        Option to specify name of raw ncdf from AMPERE. Make sure you name a file
        from the correct date.
        Default: ''

    Returns
    -------
    savefile : str
        Path to hdf-file containing Iridium data for Lompe.

    """
    '''
    Installation info: Managed to get this to run when having shapely.__version__==1.7.1 and
    astropy.__version__==4.2.1 and python==3.7

    '''
    if not basepath.endswith('/'):
        basepath += '/'
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'

    # check if the processed file exists
    savefile = tempfile_path + event.replace('-', '') + '_iridium.h5'

    if os.path.isfile(savefile):  # checks if file already exists
        return savefile

    try:
        from astropy import coordinates as coord
        from astropy import units
        from astropy.time import Time
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'read_iridium: Could not import astropy modules. Will not be able to read Iridium-files.')

    # find raw file in basepath
    if len(file_name) > 1:
        fn = basepath + file_name
    else:
        fn = basepath + event.replace('-', '') + 'Amp_invert.ncdf'
    if not os.path.isfile(fn):
        files = glob.glob(basepath + '*' + event.replace('-', '') + '*.ncdf')
        try:
            fn = files[0]
        except:
            raise FileNotFoundError(
                'Cannot find Iridium netcdf in specified folder.')  

    iridset = xr.load_dataset(fn, engine='netcdf4')

    # parse date from event string. For older AMPERE files, date was parsed from
    # the title, but that is no longer possible in the newer files.
    # datestr = ''.join([c for c in iridset.title if c.isnumeric()])
    # date = dt.datetime.strptime(datestr, '%Y%m%d').date()
    # year, month, day = date.year, date.month, date.day
    year = int(event[0:4])
    month = int(event[5:7])
    day = int(event[8:10])

    # get datetimes:
    t = iridset.time.values
    hh = np.int32(np.floor(t))
    mm = np.int32(np.floor((t - hh) * 60))
    ss = np.int32(np.floor(((t - hh) * 60 - mm) * 60))

    irid_dt = Time([dt.datetime(year, month, day, h, m, s)
                   for h, m, s in zip(hh, mm, ss)])

    # get satellite position in new coordinate system
    cart_pos = coord.CartesianRepresentation(iridset.pos_eci.values.T[0],
                                             iridset.pos_eci.values.T[1],
                                             iridset.pos_eci.values.T[2],
                                             unit=units.m)

    gcrs_pos = coord.GCRS(cart_pos, obstime=irid_dt)
    itrs_pos = gcrs_pos.transform_to(coord.ITRS(obstime=irid_dt))

    # get space mag obs in new coordinate system
    cart_B = coord.CartesianRepresentation(iridset.b_eci.values.T[0],
                                           iridset.b_eci.values.T[1],
                                           iridset.b_eci.values.T[2],
                                           unit=units.nT)
    gcrs_B = coord.GCRS(cart_B, obstime=irid_dt)
    itrs_B = gcrs_B.transform_to(coord.ITRS(obstime=irid_dt))

    lat = itrs_pos.spherical.lat.value
    lon = itrs_pos.spherical.lon.value
    r = itrs_pos.spherical.distance.value

    # convert space mag obs to east, north, up
    e = np.vstack((- np.sin(lon * d2r),
                  np.cos(lon * d2r), np.zeros_like(lon)))
    n = np.vstack((-np.sin(lat * d2r) * np.cos(lon * d2r), -
                  np.sin(lat * d2r) * np.sin(lon * d2r), np.cos(lat * d2r)))
    u = np.vstack((np.cos(lat * d2r) * np.cos(lon * d2r),
                  np.cos(lat * d2r) * np.sin(lon * d2r), np.sin(lat * d2r)))

    Bx_ecef, By_ecef, Bz_ecef = itrs_B.x.value, itrs_B.y.value, itrs_B.z.value
    Be = e[0] * Bx_ecef + e[1] * By_ecef + e[2] * Bz_ecef
    Bn = n[0] * Bx_ecef + n[1] * By_ecef + n[2] * Bz_ecef
    Bu = u[0] * Bx_ecef + u[1] * By_ecef + u[2] * Bz_ecef

    # dataframe for saving
    df = pd.DataFrame({'time': irid_dt.value, 'B_e': Be, 'B_n': Bn, 'B_r': Bu, 'B_err': iridset.b_error.values,
                       'lat': lat, 'lon': lon, 'r': r})
    df.to_hdf(savefile, key='df', mode='w')
    print('Iridium file saved: ' + savefile)

    return savefile


def getbearing(lat0, lon0, lat1, lon1):
    """
    Helper function for calculating the starting bearing angle along great circle from (lat0,lon0) to (lat1,lon1)
    input must be degrees

    """

    lat0 = lat0 * d2r
    lon0 = lon0 * d2r
    lat1 = lat1 * d2r
    lon1 = lon1 * d2r

    coslt1 = np.cos(lat1)
    sinlt1 = np.sin(lat1)
    coslt0 = np.cos(lat0)
    sinlt0 = np.sin(lat0)
    cosl0l1 = np.cos(lon1 - lon0)
    sinl0l1 = np.sin(lon1 - lon0)

    cosc = sinlt0 * sinlt1 + coslt0 * coslt1 * cosl0l1
    # Avoid roundoff problems by clamping cosine range to [-1,1].
    negs = cosc < -1.
    cosc[negs] = -1.
    poss = cosc > 1.
    cosc[poss] = 1.

    sinc = np.sqrt(1.0 - cosc**2)

    small = np.abs(sinc) > 1.0e-7  # small angle
    sinaz = np.zeros(len(lat0))
    cosaz = np.ones(len(lat0))
    cosaz[small] = (coslt0[small] * sinlt1[small] - sinlt0[small]
                    * coslt1[small] * cosl0l1[small]) / sinc[small]
    sinaz[small] = sinl0l1[small] * coslt1[small] / sinc[small]

    return np.arctan2(sinaz, cosaz)


def cross_track_los(values, glat, glon, return_bearing=True):
    """
    Helper function for calculating the line-of-sight of an instrument observing plasma drift
    in the cross-track direction

    Uses bearing angle of spacecraft trajectory (trajectory coordinates are glat, glon) and rotates
    90 degrees to cross-track direction - direction of rotation depeding on sign of measurement

    Parameters
    ----------
    values : array
        values of cross-track ion velocity measurement.
    glat : array
        geographic latitude of spacecraft location.
    glon : array
        geographic longitude of spacecraft location.
    return_bearing : bool, optional
        set to true to return bearing angle of cross-track direction. The default is True.

    Returns
    -------
    le : array
        eastward component of line-of-sight (cross track) unit vector.
    ln : array
        northward component of line-of-sight (cross track) unit vector.
    bearing : array
        bearing angle of cross-track direction. Returned when return_bearing is set to True.

    """

    # calculate bearing angle of cross track direction
    theta = getbearing(glat[:-1], glon[:-1], glat[1:],
                       glon[1:]) * r2d  # travel bearing in degrees
    theta = np.append(theta, np.nan)
    alpha = np.zeros(len(theta))      # rotate to cross-track direction

    # correcting angle for negative convection velocity
    negs = values < 0
    alpha[~negs] = 90
    alpha[negs] = -90
    bearing = theta - alpha

    negs = bearing < -180
    bearing[negs] = bearing[negs] + 360
    poss = bearing > 180
    bearing[poss] = bearing[poss] - 360
    le, ln = los_azimuth2en(bearing)  # east, north components

    if return_bearing:
        return le, ln, bearing

    return le, ln


def radar_losvec_from_mag(mlat, mlon, magazimuth, time, refh=300):
    """
    Helper function to calculate line-of-sight unit vector in geographic coords from radar file
    containing components in magnetic coordinates only. Uses Apexpy for coordinate conversion
    and vector rotation

    Parameters
    ----------
    mlat : array
        magnetic latitude of observations
    mlon : array
        magnetic longitude of observations
    magazimuth : array
        magnetic bearing angle of observation (indicating line-of-sight direction)
    time : timestamp
        determines IGRF coefficients used for conversion with Apex 
    refh : int, optional
        height of observation [km] for coordinate conversion. The default is 300.


    Returns
    -------
    glat : array
        latitude of observations (geographic)
    glon : array
        longitude of observations (geographic)
    le : array
        eastward component of line-of-sight unit vectors.
    ln : array
        northward component of line-of-sight unit vectors.
    le_m : array
        magnetic east component of line-of-sight unit vectors.
    ln_m : array
        magnetic north component of line-of-sight unit vectors.

    """

    # local imports
    try:
        import apexpy
    except ModuleNotFoundError:
        raise ModuleNotFoundError('read_sdarn requires apexpy module.')

    # line-of-sight east, north magnetic coordinates
    le_m, ln_m = los_azimuth2en(magazimuth)

    apex = apexpy.Apex(time, refh)
    glat, glon, _ = apex.apex2geo(mlat, mlon, refh)

    # find Apex base vectors to rotate los vector from magnetic to geographic
    f1, f2 = apex.basevectors_qd(glat, glon, refh, coords='geo')

    # normalize the northward vector, and define a new eastward vector that is perpendicular:
    f2 = f2 / np.linalg.norm(f2, axis=0)
    f1 = np.cross(
        np.vstack((f2, np.zeros(f2.shape[1]))).T, np.array([[0, 0, 1]])).T[:2]

    le, ln = f1 * le_m.reshape((1, -1)) + f2 * ln_m.reshape((1, -1))

    return glat, glon, le, ln, le_m, ln_m


def los_azimuth2en(azimuth):
    """
    helper function to get east, north line-of-sight unit vector from bearing angle of los direction

    Parameters
    ----------
    azimuth : array
        bearing angle of line-of-sight

    Returns
    -------
    le : array
        eastward component of line-of-sight unit vector.
    ln : array
        northward component of line-of-sight unit vector.

    """

    le, ln = np.sin(azimuth * d2r), np.cos(azimuth * d2r)

    return le, ln
