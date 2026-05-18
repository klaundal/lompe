import os
import numpy as np
import pandas as pd
import requests


def download_champ(event, basepath='./', tempfile_path='./'):
    """
    Download CHAMP data from the FTP server and process it in lompe data format.
    Note that CHAMP data is only available for the year between 2000 and 2010.

    Args:
        event (str): format 'YYYY-MM-DD'
        basepath (str, optional): path. Defaults to './'.
        tempfile_path (str, optional): path. Defaults to './'.

    Returns:
        savedfile: file name of the processed file if successful
    """
    event_date = event.replace('-', '')
    year = event[:4]
    savefile = tempfile_path + f'CH_ME_MAG_LR_3_{event_date}_0102.cdf'
    processed_file = tempfile_path + f'{event_date}_champ.h5'

    # Check if the processed file already exists
    if os.path.isfile(processed_file):
        return processed_file

    # Check if the raw file already exists
    if not os.path.isfile(savefile):
        from requests_ftp import ftp
        session = requests.Session()
        session.mount('ftp://', ftp.FTPAdapter())
        ftp_url = f"ftp://isdcftp.gfz-potsdam.de/champ/ME/Level3/MAG/V0102/{year}/CH_ME_MAG_LR_3_{event_date}_0102.cdf"
        try:
            # Downloading the file and checking if it was successful
            response = session.get(ftp_url)
            if response.status_code == 200:
                with open(savefile, "wb") as file:
                    file.write(response.content)
                print(f"Downloading {savefile} is successful!")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(
                f"No champ data: Champ data available (2000-07-19 to 2010-09-17), returning: {e}")
            return None

    # Process the downloaded CDF file ot get the magnetic disturbance
    try:
        import cdflib
        import ppigrf
        cdf_file = cdflib.CDF(savefile)
        mag = cdf_file.varget('B_NEC')  # space magnetometer data

        # geocentric coords of CHAMP orbit
        theta = 90 - cdf_file.varget('Latitude')
        phi = cdf_file.varget('Longitude')
        r = cdf_file.varget('Radius') / 1000

        time = cdflib.cdfepoch.to_datetime(cdf_file.varget('Timestamp'))

        # using IGRF to calculate magnetic disturbance (dB) registered by CHAMP
        Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, time[0])
        B0 = np.vstack((-Btheta.flatten(), Bphi.flatten(), -Br.flatten()))
        dB = mag.T - B0

        champ_df = pd.DataFrame({
            'Be': dB[1],
            'Bn': dB[0],
            'Bu': -dB[2],
            'lon': phi,
            'lat': 90 - theta,
            'r': r
        }, index=time)
        champ_df.to_hdf(processed_file, key='df', mode='w')
        os.remove(savefile)  # remove the raw file after processing
        return processed_file
    except Exception as e:
        print(f"Failed to process the file: {e}")
        return None
