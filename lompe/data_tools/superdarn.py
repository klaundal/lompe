import os
import glob
import time
import random
import shutil
import requests
import pandas as pd
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_sdarn_file(url, save_path, pbar=None):
    # function to download a file from a URL and save it to a specific path
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)

        if pbar:
            pbar.update(min(1, pbar.total - pbar.n))
        # print(f"Downloaded successfully: {save_path}")
    else:
        return None


def download_sdarn_files(event, basepath='./', pbar=None):
    filepath = os.path.dirname(__file__)
    # file containing the URLs of the SuperDARN files (zenodo records)
    file_loc = pd.read_csv(filepath + '/../data/sdarn_2010_to_2021.csv')
    # Month mapping
    month_map = {
        'Jan': '01',
        'Feb': '02',
        'Mar': '03',
        'Apr': '04',
        'May': '05',
        'Jun': '06',
        'Jul': '07',
        'Aug': '08',
        'Sep': '09',
        'Oct': '10',
        'Nov': '11',
        'Dec': '12'
    }

    # Add numerical month to DataFrame (string format)
    file_loc['Month_Num'] = file_loc['MM'].map(month_map)
    # event_date = "2019-02-25"

    # Filter the DataFrame based on the year and month of the event date
    year = event[:4]
    month = event[5:7]

    # Filter the DataFrame
    filtered_df = file_loc[(file_loc['year'].astype(
        str) == year) & (file_loc['Month_Num'] == month)]

    # Apply function and add to DataFrame
    event_date_str = event.replace('-', '')

    # URL of the Zenodo record
    url = filtered_df['url'].tolist()[0]

    # Send a GET request to the URL
    response = requests.get(url)
    time.sleep(random.uniform(0.2, 0.5))
    soup = BeautifulSoup(response.content, "html.parser")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:

        # Find all <a> tags with class "download-file"
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if 'grid.nc' in href and event_date_str in href:
                url_to_download = 'https://zenodo.org' + href
                # print(url_to_download)
                save_path = basepath + \
                    url_to_download.split('/')[-1].split('?')[0]
                download_sdarn_file(url_to_download, save_path, pbar=pbar)
            # else:
            #     print('No file found')
    else:
        print('Failed to download the file')
    return None


def download_sdarn(event, basepath='./', tempfile_path='./'):
    # tempfile_path = '/Users/fasilkebede/Documents/LOMPE/Data/SuperDARN/'
    # event = '2019-02-25'

    savefile = tempfile_path + event.replace('-', '') + '_superdarn_grdmap.h5'
    if os.path.isfile(savefile):
        return savefile
    else:
        from lompe.data_tools.dataloader import radar_losvec_from_mag
        temp_sdarn_path = basepath + f"sdarn_files_{event.replace('-', '')}/"
        os.makedirs(temp_sdarn_path, exist_ok=True)

        with tqdm(total=100, desc=f"Downloading SuperDARN for {event}") as pbar: 

            # download_sdarn_files(event, temp_sdarn_path)
            download_sdarn_files(event, temp_sdarn_path, pbar=pbar)

            # looking for the .nc files for the event
            try:
                files = glob.glob(
                    f"{temp_sdarn_path}*{event.replace('-', '')}*.nc")
                files.sort()
                ddd = pd.DataFrame()
                for file in files:
                    sm = xr.load_dataset(file, decode_timedelta=True)
                    st_abbrev = file.split('/')[-1].split('.')[1]
                    # mjd conversion
                    mjd_epoch = pd.Timestamp('1858-11-17')
                    duration = (sm['mjd_end'] + mjd_epoch) - \
                        (sm['mjd_start'] + mjd_epoch)

                    time = (sm['mjd_start'] + mjd_epoch) + duration

                    # dff['date'] = unix_epoch + dff.mjd_start

                    temp = pd.DataFrame()

                    # in degrees AACGM
                    temp.loc[:, 'mlat'] = sm['vector.mlat'].values
                    # in degrees AACGM
                    temp.loc[:, 'mlon'] = sm['vector.mlon'].values
                    # glat, glon from lompe "radar_losvec_from_mag" is a bit different from the vector.glat and vector.glon from the data??
                    # in degrees
                    temp.loc[:, 'vector.glat'] = sm['vector.glat'].values
                    # in degrees
                    temp.loc[:, 'vector.glon'] = sm['vector.glon'].values
                    # in degrees, the angle between los and magnetic north
                    temp.loc[:, 'azimuth'] = sm['vector.kvect'].values
                    # in m/s
                    temp.loc[:, 'vlos'] = sm['vector.vel.median'].values
                    # in m/s
                    temp.loc[:, 'vlos_sd'] = sm['vector.vel.sd'].values
                    # in km
                    temp.loc[:, 'range'] = sm['vector.pwr.median'].values
                    # spectral width in m/s
                    temp.loc[:, 'wdt'] = sm['vector.wdt.median'].values
                    temp.loc[:, 'time'] = pd.to_datetime(time.values).round('s')
                    temp.loc[:, 'radar'] = st_abbrev
                    # temp.set_index = pd.to_datetime(temp['time'], unit='s')
                    # dff['datetime'] = mjd_epoch + dff['mjd_start']
                    ddd = pd.concat([ddd, temp], ignore_index=True)
                ddd.set_index('time', inplace=True)
                ddd['glat'], ddd['glon'], ddd['le'], ddd['ln'], ddd['le_m'], ddd['ln_m'] = radar_losvec_from_mag(ddd['mlat'].values,
                                                                                                                ddd['mlon'].values, ddd['azimuth'].values, ddd.index[0])
                dd = ddd[ddd['glat'] > 0]  # restrict to northern hemisphere
                df_final = dd.sort_values(by='time')
                df_final.to_hdf(savefile, key='df', mode='w')
                # remove the temp files after processing
                shutil.rmtree(temp_sdarn_path)

                pbar.update(min(40, pbar.total - pbar.n))

                print(f"SuperDARN - Download complete: {savefile}")

                return savefile
            except Exception as e:
                print(f"Failed to process the file: {e}")
                return None
