import os
import datetime as dt
import requests
from lompe.data_tools.dataloader import read_iridium
from tqdm import tqdm


def ampere_parsestart(start):
    # DO NOT EDIT THIS FUNCTION

    # internal helper function adapted from supermag_api.py

    if isinstance(start, list):
        timestring = "%4.4d-%2.2d-%2.2dT%2.2d:%2.2d" % tuple(start[0:5])
    elif isinstance(start, dt.date):
        # good to go, TBD
        timestring = start.strftime("%Y-%m-%dT%H:%M")
    else:
        # is a string, reparse, TBD
        timestring = start

    return (timestring)


def ampere_coreurl(page, logon, start, extent):
    # DO NOT EDIT THIS FUNCTION

    # internal helper function adapted from supermag_api.py
    baseurl = "https://ampere.jhuapl.edu/"

    mytime = ampere_parsestart(start)
    urlstr = baseurl + 'services/' + page + '?'
    urlstr += '&logon=' + logon
    urlstr += '&start=' + mytime

    urlstr += '&extent=' + ("%12.12d" % extent)

    return (urlstr)


def download_iridium_raw(event, basepath='./'):
    """Download netcdf (dB raw) data to be used by lompe from the AMPERE database (jhuapl) for a given event
    returns an input for the lompe read_iridium script in dataloader.py in data_tools
    Example usage:
        event = '2012-04-05'
        basepath = 'downloads'
        download_iridium(event, basepath)

    Args:
        event (str): fromat YYYY-MM-DD
        basepath (str, optional): path to . Defaults to './'.
        tempfile_path (str, optional): path to. Defaults to './'.
        file_name (str, optional):name of the file to write the netcdf file. Defaults to ''.

    Returns:
        saved file: to be used by the lompe read_iridium function in data_tools

    Note: 
        functions "ampere_parsestart" and "ampere_coreurl" are internal helper functions adapted from supermag_api.py
        credit to the original author of the functions.
    """
    if not basepath.endswith('/'):
        basepath += '/'

    start = event + 'T00:00:00'
    duration = 86400  # Duration in seconds (one day)
    # check if the processed file exists
    savefile = basepath + event.replace('-', '') + '_iridium.nc'

    # checks if file already exists
    # checking if the file is not empty
    if os.path.isfile(savefile) and os.path.getsize(savefile) > 0:
        print(f"File {savefile} already exists at {basepath}.")
        return savefile
        # return read_iridium(event, basepath='./', tempfile_path='./', file_name='')
    else:
        import certifi
        # URL to download data from (lompe username is already registered in the API)
        urlstr = ampere_coreurl('data-rawdB.php', 'lompe', start, duration)
        # headers = {"User-Agent": "Mozilla/5.0"}
        # verify=certifi.where())
        response = requests.get(
            urlstr, verify=certifi.where(), stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the downloaded data to a file
            with open(savefile, 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to retrieve data: {response.status_code}")
        return savefile


def download_iridium(event, basepath='./', tempfile_path='./'):
    """Download and process iridium data for a given event, returns an input for the lompe read_iridium script in dataloader.py in data_tools
    Example usage:
        event = '2012-04-05'
        basepath = 'downloads'
        tempfile_path = 'downloads'
        file_name = '20120405_iridium.h5'
        download_iridium(event, basepath, tempfile_path, file_name)

    Args:
        event (str): fromat YYYY-MM-DD
        basepath (str, optional): path to . Defaults to './'.
        tempfile_path (str, optional): path to. Defaults to './'.
        file_name (str, optional):name of the file to write the netcdf file. Defaults to ''.

    Returns:
        saved file: to be used by the lompe read_iridium function in data_tools

    Note: 
        functions "ampere_parsestart" and "ampere_coreurl" are internal helper functions adapted from supermag_api.py
        credit to the original author of the functions.
    """
    if not basepath.endswith('/'):
        basepath += '/'
    if not tempfile_path.endswith('/'):
        tempfile_path += '/'
    savefile = tempfile_path + event.replace('-', '') + '_iridium.h5'
    raw_file_name = basepath + event.replace('-', '') + '_iridium.nc'
    if os.path.isfile(savefile) and os.path.getsize(savefile) > 0:
        print(f"Iridium/AMPERE file already exists at {savefile}.")
        return savefile
    elif os.path.isfile(raw_file_name) and os.path.getsize(raw_file_name) > 0:
        print(f"Iridium/AMPERE raw file (.nc) exists - converting to lompe data as {savefile}.")
        with tqdm(total=100, desc=f"Downloading Iridium/AMPERE for {event}") as pbar: 
            result = read_iridium(event, basepath=basepath, tempfile_path=tempfile_path, pbar=pbar)
        return result
    else:
        # print(f"File {savefile} does not exist at {tempfile_path}. Downloading raw data and converting to lompe data as {savefile}.")
        with tqdm(total=100, desc=f"Downloading Iridium/AMPERE for {event}") as pbar: 
            _ = download_iridium_raw(event, basepath=basepath)
            pbar.update(40)
            result = read_iridium(event, basepath=basepath, tempfile_path=tempfile_path, pbar=pbar)

        return result
