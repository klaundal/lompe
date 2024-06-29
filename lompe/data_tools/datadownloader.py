import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import datetime as dt
import os


def download_sussi(event, destination='downloads', source='jhuapl'):
    """function to download SSUSI data for a given event date
    Example usage:
        event = '2014-08-20'
        detination = 'downloads'
        source = 'cdaweb'
        download_sussi(event, destination, source)

    Args:
        event (str): format YYYY-MM-DD
        destination (str, optional): where to save the data. Defaults to 'downloads'.
        source (str, optional): Defaults to 'jhuapl'. cdaweb is the other option.

    Note: I (Fasil) prefer the cdaweb because it is more faster to downlaod.
          but the read_sussi function in lompe package is tailored to the jhuapl data.
    """

    year = int(event[0:4])
    doy = date2doy(event)
    os.makedirs(destination, exist_ok=True)
    # iterate over the satellites with SSUSI data
    for sat in [16, 17, 18, 19]:
        if source == 'jhuapl':
            url = f"https://ssusi.jhuapl.edu/data_retriver?spc=f{sat}&type=edr-aur&year={year}&Doy={doy}"
        elif source == 'cdaweb':
            url = f'https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/dmspf{sat}/ssusi/data/edr-aurora/{year}/{doy}/'
        else:
            print(f"Unsupported source: {source}")
            continue
        # content of the webpage
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href.lower().endswith('.nc'):  # looking for .NC (jhuapl) and .nc (cdaweb) files
                file_url = urljoin(url, href)
                try:
                    # Download the file
                    # probably too much print uncommented the print statement if needed
                    # print(f"Downloading {file_url}...")
                    response = requests.get(file_url, stream=True)
                    response.raise_for_status()  # raise an HTTPError on bad response
                    filename = os.path.join(
                        destination, os.path.basename(file_url))

                    with open(filename, 'wb') as file:
                        # 64KB chuck size
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:  # Filter out keep-alive new chunks
                                file.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {file_url}: {e}")

    print("Download complete!")
    return None


def download_smag():
    pass


def download_iridium():
    pass


def download_supermag():
    pass


def download_dmsp():
    pass


def download_swarm():
    pass

def date2doy(date_str):
    date = dt.strptime(date_str, "%Y-%m-%d")
    return date.timetuple().tm_yday


if __name__ == '__main__':
    print("This is a module to download SSUSI data for a given event date.")
