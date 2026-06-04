import os
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm


def download_swarm_mag(event, tempfile_path='./'):
    """Download Swarm data for a given event date.

    Args:
        event (str): Event date in 'YYYY-MM-DD' format.
        tempfile_path (str, optional): Path to save the file or check if it already exists. Defaults to './'.

    Returns:
        str: File name of the downloaded file if successful.

    Note:
        1. Install viresclient using "pip install viresclient" or make sure that it is installed.
        2. The request needs an access token from the Swarm website. Visit https://viresclient.readthedocs.io/en/latest/config_details.html.
    """

    savefile = tempfile_path + event.replace('-', '') + '_swarm_mag.h5'

    if os.path.isfile(savefile):
        print(f"Swarm MAG file already exists at {savefile}")
        return savefile

    try:
        from viresclient import SwarmRequest
    except ModuleNotFoundError:
        print('Please install viresclient using "pip install viresclient"')
        return
    # checking if token is present if not directing to the website to configure it
    try:
        with open(os.path.expanduser('~/.viresclient.ini'), 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("token ="):
                token_value = line.split('=', 1)[1].strip()
                # if token_value:
                #     print("Swarm token is present:", token_value)
    except:
        print("Token is missing or empty. \nPlease visit https://viresclient.readthedocs.io/en/latest/config_details.html to configure it")
        return
    try:
        request = SwarmRequest()
        event_start = dt.datetime.strptime(event, '%Y-%m-%d')
        event_end = event_start + \
            dt.timedelta(hours=23, minutes=59, seconds=59)
        df = pd.DataFrame()

        satellites = ['A', 'B', 'C']
        for swarm_satellite in tqdm(satellites, desc=f"Downloading Swarm MAG data for {event}"):
        # for swarm_satellite in ['A', 'B', 'C']:
            request.set_collection(f"SW_OPER_MAG{swarm_satellite}_LR_1B")
            request.set_products(
                measurements=["F", "B_NEC"],
                models=["CHAOS"],
                sampling_step="PT5S",  # 5 seconds if needed PT1M for 1 minute
                # quasi-dipole latitude and longitude
                auxiliaries=["MLT", "OrbitNumber", 'QDLat', 'QDLon']
            )
            data = request.get_between(
                start_time=event_start,
                end_time=event_end,
                show_progress=False
            )
            
            df = pd.concat([df, data.as_dataframe(expand=True)])

        # Removing the background field from CHAOS
        df['B_n'] = df['B_NEC_N'] - df['B_NEC_CHAOS_N']
        df['B_e'] = df['B_NEC_E'] - df['B_NEC_CHAOS_E']
        # Upward is negative in the NEC systema
        df['B_u'] = -(df['B_NEC_C'] - df['B_NEC_CHAOS_C'])

        df.sort_values(by='Timestamp', inplace=True)
        # df.reset_index(inplace=True)
        df.to_hdf(savefile, key='df', mode='w')

        print(f"Swarm MAG - Download complete: {savefile}")
        
        return savefile

    except Exception as e:
        print(f"An error occurred while processing the Swarm data: {e}")


def download_swarm_efi(event, tempfile_path = './', only_good = True):
    """
    Download Swarm EFI horizontal cross-track ion flow measurements.

    only_good = True applies strict (I think) quality flags and calibration flag
    """

    savefile = tempfile_path + event.replace('-', '') + '_swarm_efi_tct.h5'

    if os.path.isfile(savefile):
        print(f"Swarm EFI TCT file already exists at {savefile}.")
        return savefile

    try:
        from viresclient import SwarmRequest
    except ModuleNotFoundError:
        print('Please install viresclient using "pip install viresclient"')
        return

    try:
        request = SwarmRequest()
        event_start = dt.datetime.strptime(event, '%Y-%m-%d')
        event_end = event_start + dt.timedelta(hours=23, minutes=59, seconds=59)
        df = pd.DataFrame()

        satellites = ['A', 'B', 'C']
        measurements = ["Viy", "VsatE", "VsatN", "Quality_flags", "Calibration_flags"]

        for swarm_satellite in tqdm(satellites, desc=f"Downloading Swarm EFI TCT data for {event}"):
            request.set_collection(f"SW_EXPT_EFI{swarm_satellite}_TCT02")
            request.set_products(measurements = measurements)
            data = request.get_between(start_time = event_start, end_time = event_end, show_progress = False )
            df = pd.concat([df, data.as_dataframe()])

        horizontal_speed = np.hypot(df['VsatE'], df['VsatN'])
        df['le'] =  df['VsatN'] / horizontal_speed
        df['ln'] = -df['VsatE'] / horizontal_speed

        # quality and calibration flags:
        quality_flags = df['Quality_flags'].fillna(0).astype('uint16').to_numpy()
        calibration_flags = df['Calibration_flags'].fillna(0xffffffff).astype('uint32').to_numpy()
        good_quality = (quality_flags & 4) != 0
        good_calibration = ((calibration_flags >> 16) & 0xff) == 0
        good_data = good_calibration & np.isfinite(df['Viy'].to_numpy())

        good_data = good_data & good_quality

        if only_good:
            df = df[good_data]

        columns = ["Spacecraft", "Latitude", "Longitude", "Radius", "Viy", "le", "ln", "Quality_flags", "Calibration_flags"]
        df = df[[column for column in columns if column in df.columns]]
        df.sort_values(by = 'Timestamp', inplace = True)
        df.to_hdf(savefile, key = 'df', mode = 'w')

        print(f"Swarm EFI - Download complete: {savefile}")

        return savefile

    except Exception as e:
        print(f"An error occurred while processing the Swarm EFI TCT data: {e}")
