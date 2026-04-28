import warnings
import xarray as xr
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime as dt
import os
import glob
import shutil
from lompe.utils.time import date2doy
import random
import time

warnings.filterwarnings("ignore")

# extensive list of functions (including helper functions) to download data from different sources for a given event date
# the functions are designed to be used in the lompe package for data loading and processing
# this has to be integrated with the dataloader.py in the lompe package at some level to get lompe data format

# the idea is to merge the two files (datadownloader.py and dataloader.py) into one file in the lompe package, for the time being we need to keep
# both, some functions imported from dataloader.py are in use in this script

# Note that: lazy importing is implemented here


def download_ssusi(event, hemi='north', basepath='./ssusi_tempfiles', tempfile_path='./', source='cdaweb'):
    """

    Called and used for modelling auroral conductance in cmodel.py.

    Extract the relevant SSUSI info from netcdf-files downloaded from APL server.
    https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/ (ssusi/data/edr-aurora)
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
        Default: './ssusi_tempfiles'
    tempfile_path : str, optional
        Location for storing processed SSUSI data.
        Default: './'
    source : str, optional
        Specify source of SUSSI data. 
        Default: 'cdaweb' 

    Returns
    -------
    savefile : str
        Path to saved file containing SSUSI data + conductances extracted from the images.

    """
    '''
    Using the CDAWeb server here eliminates the need for checking if the data is from the same day or not.
    The data is already sorted by date and time. From the APL sever, there might be a need to check if the data is from the same day.
    (see download_ssusi function)
    
    Extract the relevant info from downloaded netcdf-files (similar to from APL server, but from CDAWeb):
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
        print('SSUSI file already exists.')
        return savefile
    try:
        import netCDF4
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'read_ssusi: Could not load netCDF4 module. Will not be able to read SSUSI-files from APL.')
    # downlaoding the files from the server see the ssusi_direct.py for the implementation
    # from lompe.data_tools.ssusi_direct2 import download_ssusi_files
    download_ssusi_files(event, basepath=basepath, source=source)
    ###
    imgs = []  # List to hold all images. Converted to xarray later.
    doy_str = f"{date2doy(event):03d}"
    for sat in ['F16', 'F17', 'F18', 'F19']:
        files = glob.glob(basepath + '*' + sat.lower() + '*' +
                          event[0:4] + doy_str + '*.nc')
        files.sort()
        if len(files) == 0:
            continue

        ii = 0  # counter

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
                    {'orbit': file.split('-')[-1].split('_')[0][3:]})
                imgs.append(img)
            ii += 1

    if len(imgs) == 0:
        print('No SSUSI images found.')

    else:     # save as netcdf in specified path
        imgs = xr.concat(imgs, dim='date')
        imgs = imgs.assign({'hemisphere': hemi})
        imgs = imgs.sortby(imgs['date'])
        print('DMSP SSUSI file saved: ' + savefile)

        imgs.to_netcdf(savefile)
        shutil.rmtree(basepath)
        return savefile


def fetch_ssusi_urls(sat, year, doy, source):
    doy_str = f"{doy:03d}"  # Zero-pad the day of year to three digits
    # if source == 'jhuapl':
    #     url = f"https://ssusi.jhuapl.edu/data_retriver?spc=f{sat}&type=edr-aur&year={year}&Doy={doy_str}"
    if source == 'cdaweb':
        url = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf{sat}/ssusi/data/edr-aurora/{year}/{doy_str}/'
    else:
        print(f"Unsupported source: {source}")
        return []

    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an HTTPError on bad response
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch URL probably no data : \n{e}")
        return []

    from urllib.parse import urljoin
    soup = BeautifulSoup(response.content, 'html.parser')
    urls = [urljoin(url, link.get('href')) for link in soup.find_all(
        'a', href=True) if link.get('href').lower().endswith('.nc')]

    if not urls:
        print(
            f"No .nc files found for satellite F{sat} on day {doy_str} of year {year} from source {source}")

    return urls


def download_ssusi_file(file_url, destination):
    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # raise an HTTPError on bad response
        filename = os.path.join(destination, os.path.basename(file_url))

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_url}: {e}")


def download_ssusi_files(event, source='cedaweb', basepath='./ssusi_tempfiles/'):
    """Downloading data from the SSUSI instrument onboard the DMSP satellites for a given event date.

    Args:
        event strin: format 'YYYY-MM-DD'
        source (str, optional): data source. Defaults to 'cedaweb'. 
    """
    year = int(event[0:4])
    doy = date2doy(event)

    os.makedirs(basepath, exist_ok=True)

    from joblib import Parallel, delayed
    import itertools
    # Use joblib.Parallel to fetch URLs concurrently
    fetch_args = [(sat, year, doy, source) for sat in [16, 17, 18, 19]]
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(fetch_ssusi_urls)(*args) for args in fetch_args)
    # results = Parallel(n_jobs=-1)(delayed(fetch_ssusi_urls)(*args) for args in fetch_args)
    all_urls = list(itertools.chain.from_iterable(results))

    # Use joblib.Parallel to download files concurrently
    download_args = [(url, basepath) for url in all_urls]
    Parallel(n_jobs=-1, backend='threading')(
        delayed(download_ssusi_file)(*args) for args in download_args)
