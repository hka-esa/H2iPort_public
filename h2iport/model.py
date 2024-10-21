import logging
import warnings
from pathlib import Path

import draf
import matplotlib.cbook

from h2iport.components import (
    BES,
    DHN,
    EG,
    EHP,
    HDS,
    PV,
    TES,
    WT,
    Con,
    DCon,
    Dem,
    Elc,
    Lan,
    Main,
    OnPV,
    Tra,
)
from h2iport.config import Config

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
logging.getLogger("draf.core.case_study").setLevel(level=logging.INFO)
logging.getLogger("draf.core.scenario").setLevel(level=logging.INFO)


def main():
    cs_name = Path(__file__).stem
    coords = tuple(Config.cf.main.coords)

    cs = draf.CaseStudy(cs_name, year=2019, freq="60min", coords=coords, consider_invest=True)
    cs.set_time_horizon(start="Mar-1 00:00", steps=cs.steps_per_day * 10)

    sc = cs.add_REF_scen(
        components=[BES, Con, DCon, Dem, DHN, EG, EHP, Elc, HDS, Lan, Main, OnPV, PV, TES, Tra, WT]
    )
