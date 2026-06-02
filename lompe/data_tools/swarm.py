import os
import datetime as dt
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
        print(f"Swarm MAG file already exists at {savefile}.")
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
                sampling_step="PT10S",  # 10 seconds if needed PT1M for 1 minute
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
