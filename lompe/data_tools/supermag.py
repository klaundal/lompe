from joblib import Parallel, delayed
import numpy as np
import os
import time
from urllib.request import urlopen
import json
import pandas as pd
from urllib.error import URLError, HTTPError
from tqdm import tqdm
# start_date = '2019-10-15T00:00:00'
# extent = 86400  # one hour (86400 for one day)


def get_smag_stations(start_date, extent, userid="lompe"):
    """ inventory of stations that have data for the given time range

    Args:
        start_date (str): eg. '2019-10-15T00:00:00'
        extent (int): eg. 86400 for one day
        userid (str, optional): _description_. Defaults to "lompe". plese use your own userid if you have one (please register if not), to avoid hitting SuperMAG limits.

    Returns:
        list: list of IAGA station codes (e.g. ['AAA', 'ABK', ...])
    """
    urlstr = f"https://supermag.jhuapl.edu/services/inventory.php?fmt=json&logon={userid}&start={start_date}&extent={extent}"
    with urlopen(urlstr) as response:
        data_response = response.read()
        data = data_response.decode('utf-8').split('\n')

    # removing ('OK', empty strings)
    json_parts = [x for x in data if x.strip() not in ("OK", "")]
    # the first part is the number of stations, so we skip it
    return json_parts[1:]


def get_smag_station_data(start_date, extent, station, userid="lompe", retries=5, sleep=2):
    """_summary_

    Args:
        start_date (str): eg. '2019-10-15T00:00:00'
        extent (int): eg. 86400 for one day
        station (str): eg. 'ABK'
        userid (str, optional): _description_. Defaults to "lompe".
        retries (int, optional): _description_. Defaults to 5.
        sleep (int, optional): _description_. Defaults to 2.

    Raises:
        ValueError: _description_

    Returns:
        pandas dataframe: dataframe with columns ['tval', 'glat', 'glon', 'N.nez', 'E.nez', 'Z.nez', 'station'] and datetime index
    """
    data_url = (
        "https://supermag.jhuapl.edu/services/data-api.php"
        f"?fmt=json&logon={userid}&start={start_date}&extent={extent}"
        f"&geo&station={station}"
    )

    last_error = None

    for attempt in range(retries):
        try:
            with urlopen(data_url, timeout=60) as response:
                text = response.read().decode("utf-8").strip()

            lines = text.splitlines()
            json_parts = [x for x in lines if x.strip() not in ("OK", "")]
            json_str = "".join(json_parts).strip()

            if not json_str:
                raise ValueError(f"Empty response for station {station}")

            parsed = json.loads(json_str)

            df = (
                pd.json_normalize(parsed)
                .assign(datetime=lambda x: pd.to_datetime(x["tval"], unit="s", utc=True))
                .set_index("datetime")
            )

            df["station"] = station
            return df

        except (json.JSONDecodeError, ValueError, HTTPError, URLError, TimeoutError) as e:
            last_error = e
            time.sleep(sleep * (attempt + 1))

    print(f"Skipping station {station}: {last_error}")
    return pd.DataFrame()


def download_supermag(event, userid="lompe", n_jobs=-1, save=True, tempfile_path="./"):
    """_summary_

    Args:
        event (_type_): _description_
        userid (str, optional): Defaults to "lompe".
        n_jobs (int, optional): Defaults to -1.
        save (bool, optional): Defaults to False.
        tempfile_path (str, optional): Path to check if the file already exists (to avoid downloading it again) Defaults to "./".

    Raises:
        RuntimeError: _description_

    Returns:
        if save=true: event_supermag.h5 file
        if save=false: pandas dataframe with columns ['Be', 'Bn', 'Bu', 'lat', 'lon', 'station'] and datetime index
    """
    start = event + "T00:00:00"
    extent = 86400
    savefile = os.path.join(
        tempfile_path, event.replace("-", "") + "_supermag.h5")

    if save and os.path.isfile(savefile):
        print(f"SuperMAG file for {event} already exists at {savefile}")
        return savefile
    
    stations = get_smag_stations(start, extent, userid=userid)

    dfs = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(get_smag_station_data)(start, extent, station, userid)
        for station in tqdm(stations, desc=f"Downloading SuperMAG for {event}")
    )

    # remove failed / empty downloads
    dfs = [df for df in dfs if df is not None and not df.empty]

    if not dfs:
        raise RuntimeError("No station data downloaded successfully.")

    # combine all stations into one dataframe and replace SuperMAG bad values with NaN
    df_combined = pd.concat(dfs, axis=0).sort_index()
    df_combined = df_combined.replace(999999.0, np.nan)

    # final Lompe-style dataframe (taking only the nez components and renaming columns)
    df_final = (
        df_combined
        .rename(columns={
            "glat": "lat",
            "glon": "lon",
            "N.nez": "Bn",
            "E.nez": "Be",
            "Z.nez": "Bu",
        })
        [["Be", "Bn", "Bu", "lat", "lon", "station"]]
        .dropna()
        .sort_index()
    )

    # SuperMAG Z is positive downward; Lompe wants upward
    df_final["Bu"] = -df_final["Bu"]

    print(f"SuperMAG - Download complete: {savefile}")

        if save:
            df_final.to_hdf(savefile, key="df_final", mode="w")
            print(f"SuperMAG download complete: {savefile}")
            return savefile

        return df_final

    except Exception as e:
        print("SuperMAG download failed after all tries. Please run it again.")
        print(type(e).__name__, ":", e)
        return None


def _get_supermag_inventory(userid, start, extent=3600, retries=5, backoff_factor=0.5):
    last_error = None

    for attempt in range(retries):
        try:
            status, stations = SuperMAGGetInventory(userid, start, extent)

            if stations is None or len(stations) == 0:
                raise ValueError("No stations returned")

            return stations

        except Exception as e:
            last_error = e
            wait = backoff_factor * (2 ** attempt)
            time.sleep(wait + random.uniform(0, 0.5))

    raise RuntimeError(
        f"SuperMAG inventory failed after {retries} attempts. Last error: {last_error}"
    )


def _get_supermag_station_data(
    userid,
    start,
    station,
    extent=86400,
    retries=5,
    backoff_factor=0.5,
):
    def is_valid_data(data):
        if not isinstance(data, pd.DataFrame):
            return False
        if data.empty:
            return False

        required = {"tval", "N", "E", "Z"}
        if not required.issubset(data.columns):
            return False

        first_cell = str(data.iloc[0, 0])
        bad_patterns = [
            "<br",
            "Warning",
            "shell_exec",
            "Fatal error",
            "Notice:",
            "Undefined",
        ]

        if any(pattern in first_cell for pattern in bad_patterns):
            return False

        for col in ["N", "E", "Z"]:
            vals = data[col].dropna()
            if vals.empty:
                return False

            first_val = vals.iloc[0]
            if not isinstance(first_val, dict):
                return False
            if "geo" not in first_val:
                return False

        return True

    for attempt in range(retries):
        try:
            status, data = SuperMAGGetData(
                userid,
                start,
                extent,
                "geo",
                station,
            )

            if not is_valid_data(data):
                raise ValueError("Invalid or corrupted SuperMAG response")

            data["N_geo"] = data["N"].apply(lambda x: x["geo"])
            data["E_geo"] = data["E"].apply(lambda x: x["geo"])
            data["Z_geo"] = data["Z"].apply(lambda x: x["geo"])

            if "nez" in data["N"].dropna().iloc[0]:
                data["N_nez"] = data["N"].apply(lambda x: x["nez"])
                data["E_nez"] = data["E"].apply(lambda x: x["nez"])
                data["Z_nez"] = data["Z"].apply(lambda x: x["nez"])

            data = (
                data.drop(columns=["N", "E", "Z"])
                .assign(
                    datetime=lambda df: pd.to_datetime(
                        df["tval"],
                        unit="s",
                        origin="unix",
                        utc=True,
                    ),
                    station=station,
                )
                .drop(columns="tval")
                .set_index("datetime")
            )

            return data

        except Exception:
            wait = backoff_factor * (2 ** attempt)
            time.sleep(wait + random.uniform(0, 0.5))

    return None


def _get_supermag_all_stations(
    userid,
    start,
    stations=None,
    inventory_extent=3600,
    data_extent=86400,
    max_workers=5,
    retries=5,
    backoff_factor=0.5,
    show_progress=True,
):
    if stations is None:
        stations = _get_supermag_inventory(
            userid=userid,
            start=start,
            extent=inventory_extent,
            retries=retries,
            backoff_factor=backoff_factor,
        )
        filepath = os.path.dirname(__file__)
        smag_stations = pd.read_csv(
            filepath + '/../data/supermag_stations.csv', engine="python", on_bad_lines=lambda row: row[:7] + [",".join(row[7:])])
        high_lat = smag_stations[smag_stations["GEOLAT"] >= 50]
        stations = np.intersect1d(stations, high_lat["IAGA"].values)

    results = []
    failed = []

    def fetch_station(station):
        try:
            df = _get_supermag_station_data(
                userid,
                start,
                station,
                data_extent,
                retries,
                backoff_factor,
            )
            return station, df
        except Exception:
            return station, None

    iterator = Parallel(
        n_jobs=max_workers,
        prefer="threads",
        return_as="generator_unordered",
    )(delayed(fetch_station)(station) for station in stations)

    # event = f"{start[0]:04d}-{start[1]:02d}-{start[2]:02d}"
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(stations),
            desc=f"Retrieving SuperMAG data for {start}",
            unit="station",
        )

    for station, df in iterator:
        if df is None:
            failed.append(station)
        else:
            results.append(df)

    if results:
        all_data = pd.concat(results, axis=0).sort_index()
    else:
        all_data = pd.DataFrame()

    return all_data, failed
