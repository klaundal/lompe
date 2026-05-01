import os
import datetime as dt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_dmsp_ssies(event, sat, tempfile_path='./', **madrigal_kwargs):
    """ Download DMSP SSIES ion drift meter data for a full day
        using FTP-like acess at http://cedar.openmadrigal.org/ftp/ and get relevant parameters for Lompe. 
        Saves hdf file in tempfile_path and returns path

    Args:
        event (str): 
            string on format 'yyyy-mm-dd' to specify time of event for model.
        sat (int): 
            Satellite ID for DMSP Block 5D satellite (16-19)
        tempfile_path (str, optional): 
            Path to dir where processed hdf files are placed. Default: './'
        **madrigalkwargs (dict): 
            needed to download data from MadrigalWeb (Madrigal needs user specifications through **madrigal_kwargs.)
    Example usage:
        event = '2014-07-13'
        sat = 17
        tempfile_path = '/Users/fasilkebede/Documents/'
        madrigal_kwargs = {'user_fullname': 'First','user_email': 'name@host.com', 'user_affiliation': 'University'}

    Returns:
        savefile(str):
            Path to hdf-file containing SSIES data for Lompe.
    """

    savefile = tempfile_path + \
        event.replace('-', '') + '_ssies_f' + str(sat) + '.h5'
    if os.path.exists(savefile):
        print(f'DMSP/SSIES F{sat} file already exists at {savefile}')
        return savefile

    date_str = event.replace('-', '')
    year = event[:4]
    url_base = "https://cedar.openmadrigal.org"
    url = url_base + \
        f"/ftp/fullname/{madrigal_kwargs['user_fullname']}/email/{madrigal_kwargs['user_email']}/affiliation/{madrigal_kwargs['user_affiliation']}/kinst/8100/year/{year}/"

    pbar = tqdm(total=100, desc=f"Downloading DMSP/SSIES F{sat} for {event}")

    response2 = requests.get(url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    urls = [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/' in link.get('href')]

    url_ion_drift = url_base + [a['href'] for a in soup2.find_all(
        'a', href=True) if 'ion drift' in a.text and f'F{sat}' in a.text][0]
    url_plasma_temp = url_base + [a['href'] for a in soup2.find_all(
        'a', href=True) if 'plasma temp' in a.text and f'F{sat}' in a.text][0]
    # url_flux_energy = url_base + [a['href'] for a in soup2.find_all(
    #     'a', href=True) if 'flux/energy' in a.text and f'F{sat}' in a.text][0]

    pbar.update(10)

    # downloading the ion drift file
    ion_drift_hdf5_file_url = url_ion_drift + 'format/hdf5/'
    response2 = requests.get(ion_drift_hdf5_file_url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    url_target_file = url_base + [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/fullFilename/' in link.get('href') and date_str in link.get('href')][0]

    response = requests.get(url_target_file, stream=True)
    # tempfile_path = '/Users/fasilkebede/Documents/'
    filename = tempfile_path + url_target_file[-27:-1]
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)

    pbar.update(35)
    
    # downloading the plasma temperature file
    plasma_hdf5_file_url = url_plasma_temp + 'format/hdf5/'
    response2 = requests.get(plasma_hdf5_file_url)
    soup2 = BeautifulSoup(response2.content, 'html.parser')
    url_target_file = url_base + [link.get('href') for link in soup2.find_all(
        'a', href=True) if '/format/hdf5/fullFilename/' in link.get('href') and date_str in link.get('href')][0]

    response = requests.get(url_target_file, stream=True)
    # tempfile_path = '/Users/fasilkebede/Documents/'
    filename2 = tempfile_path + url_target_file[-27:-1]
    with open(filename2, 'wb') as file:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)

    pbar.update(35)

    dmsp = pd.read_hdf(filename, mode='r', key='Data/Table Layout')
    dmsp2 = pd.read_hdf(filename2, mode='r', key='Data/Table Layout')

    dmsp.index = np.arange(len(dmsp))
    dmsp2.index = np.arange(len(dmsp2))

    # set datetime as index
    times = []
    for i in range(len(dmsp)):
        times.append(dt.datetime.fromtimestamp(
            int(dmsp['ut1_unix'][i]), tz=dt.timezone.utc))
    dmsp.index = times

    # reindexing due to lower cadence measurements
    times2 = []
    for i in range(len(dmsp2)):
        times2.append(dt.datetime.fromtimestamp(
            int(dmsp2['ut1_unix'][i]), tz=dt.timezone.utc))
    dmsp2.index = times2
    dmsp2 = dmsp2.reindex(index=dmsp.index, method='nearest', tolerance='2sec')

    dmsp.loc[:, 'po+'] = dmsp2['po+']
    dmsp.loc[:, 'te'] = dmsp2['te']
    dmsp.loc[:, 'ti'] = dmsp2['ti']

    # Smooth the orbit
    import matplotlib
    from scipy import interpolate
    from lompe.data_tools.dataloader import cross_track_los

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

    # i got negative and above 360 values even negative are excluded before the interpolation, due to the interpolation, handling it here
    glon[glon < 0] += 360
    glon = np.mod(glon, 360)

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
    os.remove(filename)
    os.remove(filename2)

    pbar.update(20)

    print(f"DMSP/SSIES F{sat} - Download complete: {savefile}")

    return savefile
