import os
import numpy as np
import pandas as pd
import lompe
from lompe.data_tools import datadownloader, dataloader

from .dmsp_ssusi import download_ssusi
from .supermag import download_supermag
from .ampere import download_iridium
from .champ import download_champ
from .superdarn import download_sdarn
from .swarm import download_swarm
from .dmsp_ssies import download_dmsp_ssies


def prepare_event_data(event, data_path="./sample_dataset/", sources=None, basepath="./sample_dataset/", **kwargs):
    """
    Download and load all event datasets once.
    Returns dict with {supermag, superdarn, champ} as DataFrames.
    """
    if sources is None:
        sources = ["supermag", "iridium", "superdarn", "champ"]
    os.makedirs(
        data_path, exist_ok=True)  # if not already existing create the directory
    event = pd.to_datetime(event).strftime("%Y-%m-%d")
    # event = event_time.strftime("%Y-%m-%d")

    def safe_read_hdf(filename):
        if filename is None or not os.path.exists(filename):
            return pd.DataFrame()
        try:
            return pd.read_hdf(filename)
        except Exception as e:
            print(f"[WARN] Could not read {filename}: {e}")
            return pd.DataFrame()
    files = {}
    if "ssusi" in sources:
        files["ssusi"] = download_ssusi(event, tempfile_path=data_path)

    if "supermag" in sources:
        files["supermag"] = download_supermag(
            event, tempfile_path=data_path)

    if "iridium" in sources:
        files["iridium"] = download_iridium(
            event, basepath=basepath, tempfile_path=data_path)

    if "champ" in sources:
        files["champ"] = download_champ(event, tempfile_path=data_path)

    if "superdarn" in sources:
        files["superdarn"] = download_sdarn(event, tempfile_path=data_path)

    if "swarm" in sources:
        files["swarm"] = download_swarm(event, tempfile_path=data_path)
    # smag_file = datadownloader.download_supermag(
    #     event, tempfile_path=data_path)
    # sdarn_file = datadownloader.download_sdarn(event, tempfile_path=data_path)
    # champ_file = datadownloader.download_champ(event, tempfile_path=data_path)
    # iridium_file = datadownloader.download_iridium(
    #     event, tempfile_path=data_path)
    # file_iridium = dataloader.read_iridium(
    #     event, file_name=iridium_file, tempfile_path=data_path)
    result = {}

    for item, file in files.items():
        result[item] = safe_read_hdf(file)

    return result


def get_data_subsets(event_data, event, delta_minutes=2, sources=None, **kwargs):
    '''
    Extract data subsets for the given time interval [t0, t1]. and prepare lompe.Data objects.
    Returns: iridium_data, supermag_data, superdarn_data, champ_data'''
    if sources is None:
        sources = ["supermag", "iridium", "superdarn", "champ"]
    T0 = pd.to_datetime(event)
    DT = pd.Timedelta(minutes=delta_minutes)
    t0, t1 = T0 - DT / 2, T0 + DT / 2

    def ensure_datetimeindex(df):
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise TypeError(f"Failed to convert index to datetime: {e}")
        return df

    # --- iridium ---
    iridium = event_data["iridium"]
    irid = iridium[(iridium.time >= t0) & (iridium.time <= t1)]

    if not irid.empty:
        irid_B = np.vstack(
            (irid.B_e.values, irid.B_n.values, irid.B_r.values))
        irid_coords = np.vstack(
            (irid.lon.values, irid.lat.values, irid.r.values))
    else:
        irid_B = np.empty((3, 0))
        irid_coords = np.empty((2, 0))
    iridium_data = lompe.Data(
        irid_B * 1e-9 if irid_B.size else irid_B,
        irid_coords,
        datatype="space_mag_fac", iweight=1.0, error=30e-9
    )

    # --- SuperMAG ---
    smag_df = ensure_datetimeindex(event_data["supermag"])
    smag_df.index = smag_df.index.tz_localize(None)
    smag = smag_df.loc[t0:t1, :]

    if not smag.empty:
        smag_B = np.vstack((smag.Be.values, smag.Bn.values, smag.Bu.values))
        smag_coords = np.vstack((smag.lon.values, smag.lat.values))
    else:
        smag_B = np.empty((3, 0))
        smag_coords = np.empty((2, 0))
    supermag_data = lompe.Data(
        smag_B * 1e-9 if smag_B.size else smag_B,
        smag_coords,
        datatype="ground_mag", iweight=0.4, error=10e-9
    )

    # --- CHAMP ---
    ch_df = ensure_datetimeindex(event_data["champ"])
    ch = ch_df.loc[t0:t1, :]
    if not ch.empty:
        champ_B = np.vstack((ch.Be.values, ch.Bn.values, ch.Bu.values))
        champ_coords = np.vstack(
            (ch.lon.values, ch.lat.values, ch.r.values*1000))
    else:
        champ_B = np.empty((3, 0))
        champ_coords = np.empty((3, 0))
    cham_data = lompe.Data(
        champ_B * 1e-9 if champ_B.size else champ_B,
        champ_coords,
        datatype="space_mag_full", iweight=0.4, error=10e-9
    )

    # --- SuperDARN ---
    sd = event_data["superdarn"].loc[t0:t1, :]
    if not sd.empty:
        vlos = sd["vlos"].values
        sd_coords = np.vstack((sd["glon"].values, sd["glat"].values))
        los = np.vstack((sd["le"].values, sd["ln"].values))
    else:
        vlos = np.empty((0,))
        sd_coords = np.empty((2, 0))
        los = np.empty((2, 0))
    superdarn_data = lompe.Data(
        vlos, sd_coords, LOS=los,
        datatype="convection", iweight=1.0, error=50
    )

    return iridium_data, supermag_data, superdarn_data, cham_data


# Example usage:
if __name__ == "__main__":
    event_time = "2015-03-17 04:00"
    event_data = prepare_event_data(event_time)
    iridium_data, supermag_data, superdarn_data, cham_data = get_data_subsets(
        event_data, event_time, delta_minutes=3)

    # print(iridium_data)
    # print(supermag_data)
    # print(superdarn_data)
    # print(cham_data)
    print("Data subsets prepared successfully.")
