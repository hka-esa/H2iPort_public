from typing import Dict

import pandas as pd
from draf import helper as hp

from h2iport.config import Config
from h2iport.paths import BASE_DIR

cf = Config.cf


def make_time_series(profile_csv, annual_energy) -> pd.Series:
    fp = BASE_DIR / profile_csv
    ser = hp.read(fp)
    scale_factor = annual_energy / ser.sum()
    return ser * scale_factor


def y_DHN_avail_TH(sc, cf) -> pd.Series:
    df = pd.DataFrame(index=sc.dtindex)
    ss = f"{sc.year}-{cf.comp.DHN.start.DHN_summer}"
    ws = f"{sc.year}-{cf.comp.DHN.start.DHN_winter}"
    is_summer = (df.index > ss) & (df.index < ws)
    df["DHN_summer"] = is_summer
    df["DHN_winter"] = ~is_summer
    ser = df.reset_index(drop=True).stack().astype(int)
    return ser


def T_EHP_rhine_T(sc) -> pd.Series:
    """Returns Rhine water temperature time series from 2022 monthly average values
    read off from [1].

    [1] http://luadb.lds.nrw.de/LUA/hygon/pegel.php?stationsname_t=Bad-Honnef&yAchse=Standard&hoehe=468&breite=724&jahr=2022&jahreswerte=ok
    """
    rhine_temp = {
        1: 6.5,
        2: 6.5,
        3: 8.5,
        4: 11.5,
        5: 17.5,
        6: 22.0,
        7: 24,
        8: 24,
        9: 19,
        10: 15.5,
        11: 12,
        12: 6.5,
    }
    ser = pd.Series(index=sc.dtindex)
    for k, v in rhine_temp.items():
        ser.loc[f"{sc.year}-{k}"] = v
    ser = ser.reset_index(drop=True)
    return ser

def get_DHN_curve() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=";",index_col=0)
    ser = pd.Series(df["temp_dhn"].values)
    return ser

def get_PV_PPA_profile() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=",",index_col=0)
    ser = pd.Series(df["electricity_PV_normiert"].values)
    return ser

def get_WTon_PPA_profile() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=",",index_col=0)
    ser = pd.Series(df["electricity_Onshore_normiert"].values)
    return ser

def get_WToff_PPA_profile() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=",",index_col=0)
    ser = pd.Series(df["electricity_Offshore_normiert"].values)
    return ser

def get_WTon_Ka_profile() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=",",index_col=0)
    ser = pd.Series(df["Power"].values)
    return ser

def get_var_Price_profile() -> pd.Series:
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp, sep=",",index_col=0)
    ser = pd.Series(df["prices2019"].values)
    return ser

def dH_Dem_TYAR(sc) -> pd.Series:
    d = {
        (y, a, r): make_time_series(
            profile_csv=cf.consumer_data[a]["demand"]["profile_csv"][r],
            annual_energy=cf.consumer_data[a]["demand"]["annual_energy"][r][y],
        )
        for y in sc.dims.Y
        for a in sc.dims.A
        for r in sc.dims.R
    }
    df = pd.concat(d, 1)
    ser = df.stack([0, 1, 2])
    return ser


def get_h2_station_demand(year):
    ser = pd.Series(cf.consumer_data.H2_filling_station.daily_H2_demand_in_kg)
    df = pd.DataFrame(cf.consumer_data.H2_filling_station.number_of_relevant_stations)
    annual_energy_in_kWh = ser * df.T * cf.energy_density.H2 * 365
    total_annual_energy = annual_energy_in_kWh.T.sum()[year]
    ser = make_time_series("data/dummy/profiles/equally_distributed.csv", total_annual_energy)
    return ser

def get_lan_profile():
    fp = BASE_DIR / "data/dummy/profiles/XXX.csv"
    df = pd.read_csv(fp)
    ser = pd.Series(df["Werte"].values)
    return(ser)