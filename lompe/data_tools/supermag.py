import os
import random
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

__all__ = ["download_supermag"]

try:
    from supermag_api.supermag_api import *
except ImportError:
    raise ImportError(
        "supermag-api is not installed. Install it with:\n\n"
        "pip install supermag-api"
    )


def download_supermag(
    event,
    userid="lompe",
    tempfile_path="./",
    save=True,
    inventory_extent=3600,
    data_extent=86400,
    max_workers=5,
    retries=5,
    backoff_factor=0.5,
    max_retry_rounds=4,
):
    """Download SuperMAG geogrpahic magnetic field data for one event day.

    This is the public entry point for the module. It downloads all available
    high-latitude SuperMAG stations for the requested day, retries failed
    stations in bounded rounds, converts the output to Lompe-style columns, and
    optionally saves the result as an HDF5 file.

    Args:
        event (str): Date string in ``YYYY-MM-DD`` format.
        userid (str): SuperMAG user id.
        tempfile_path (str): Output directory if ``save=True``.
        save (bool): If True, save to HDF5 and return the filepath.
        inventory_extent (int): Inventory search window in seconds.
        data_extent (int): Station data window in seconds.
        max_workers (int): Number of joblib thread workers.
        retries (int): Retries per inventory or station request.
        backoff_factor (float): Exponential backoff base in seconds.
        max_retry_rounds (int): Maximum retry rounds over failed stations.

    Returns:
        str | pandas.DataFrame | None:
            - saved filepath if ``save=True`` and successful
            - dataframe if ``save=False`` and successful
            - ``None`` if stations still fail after all retry rounds
    """
    # tempfile_path = os.path.abspath(tempfile_path)
    os.makedirs(tempfile_path, exist_ok=True)

    savefile = os.path.join(
        tempfile_path,
        event.replace("-", "") + "_supermag.h5",
    )

    if save and os.path.isfile(savefile):
        print("File already exists at: ", savefile)
        return savefile

    start = [int(x) for x in event.split("-")] + [0, 0]

    try:
        all_data, failed = _get_supermag_all_stations(
            userid=userid,
            start=start,
            inventory_extent=inventory_extent,
            data_extent=data_extent,
            max_workers=max_workers,
            retries=retries,
            backoff_factor=backoff_factor,
            show_progress=True,
        )

        retry_round = 0
        while failed and retry_round < max_retry_rounds:
            retry_round += 1
            print(
                f"Retry round {retry_round}/{max_retry_rounds} for "
                f"{len(failed)} failed stations."
            )
            retry_data, still_failed = _get_supermag_all_stations(
                userid=userid,
                start=start,
                stations=failed,
                inventory_extent=inventory_extent,
                data_extent=data_extent,
                max_workers=max_workers,
                retries=retries,
                backoff_factor=backoff_factor,
                show_progress=True,
            )

            if not retry_data.empty:
                all_data = pd.concat(
                    [all_data, retry_data], axis=0).sort_index()

            failed = still_failed

        if failed:
            print(
                f"Some stations failed after {max_retry_rounds} retry rounds. "
                "Please run it again."
            )
            return None

        df_final = (
            all_data.rename(
                columns={
                    "glat": "lat",
                    "glon": "lon",
                    "N_geo": "Bn",
                    "E_geo": "Be",
                    "Z_geo": "Bu",
                }
            )[["Be", "Bn", "Bu", "lat", "lon", "station"]]
            .dropna()
            .sort_index()
        )

        # SuperMAG Z is positive downward; Lompe wants upward.
        df_final["Bu"] = -df_final["Bu"]

        if save:
            df_final.to_hdf(savefile, key="df_final", mode="w")
            print(f"Success: saved to {savefile}")
            return savefile

        print("Success")
        return df_final

    except Exception:
        print("Download failed after all tries. Please run it again.")
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
        f"Inventory failed after {retries} attempts. Last error: {last_error}"
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
        smag_stations = pd.read_csv(
            "/Users/fasilkebede/Documents/LOMPE/substorm/supermag_stations_info.csv"
        )
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

    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(stations),
            desc="Downloading SuperMAG",
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
