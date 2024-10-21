import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from draf import Collectors, Dimensions, Params, Results, Scenario, Vars
from draf import helper as hp
from draf.abstract_component import Component
from draf.conventions import Descs, Etypes
from draf.helper import conv, get_annuity_factor, set_component_order_by_order_restrictions
from draf.paths import DATA_DIR
from draf.prep import DataBase as db
from gurobipy import GRB, Model, quicksum

from h2iport import prep, util
from h2iport.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)

cf = Config.cf


@dataclass
class Main(Component):
    """Objective functions and general collectors. This must be the last model_func to be executed."""

    def dim_func(self, sc: Scenario):
        sc.dim("Y", data=cf.projections.years, doc="Investigated years")

    def param_func(self, sc: Scenario):
        sc.collector("P_EL_source_TY", doc="Power sources", unit="kW_el")
        sc.collector("P_EL_sink_TY", doc="Power sinks", unit="kW_el")
        sc.collector("dQ_heating_source_TYH", doc="Heating energy flow sources", unit="kW_th")
        sc.collector("dQ_heating_sink_TYH", doc="Heating energy flow sinks", unit="kW_th")
        sc.collector("dQ_cooling_source_TYN", doc="Cooling energy flow sinks", unit="kW_th")
        sc.collector("dQ_cooling_sink_TYN", doc="Cooling energy flow sinks", unit="kW_th")
        sc.collector("dQ_amb_sink_TY", doc="Thermal energy flow from ambient", unit="kW_th")
        sc.collector("dQ_rhine_sink_TY", doc="Thermal energy flow from ambient", unit="kW_th")
        sc.collector("dQ_amb_source_TY", doc="Thermal energy flow to ambient", unit="kW_th")
        sc.collector("dQ_rhine_source_TY", doc="Thermal energy flow to ambient", unit="kW_th")
        sc.collector("dH_H2D_source_TYR", doc="Hub-side H2D energy flow sources", unit="kW")
        sc.collector("dH_H2D_sink_TYR", doc="Hub-side H2D energy flow sinks", unit="kW")
        sc.collector("dH_DH2D_source_TYAR", doc="Demand-side H2D energy flow sources", unit="kW")
        sc.collector("dH_DH2D_sink_TYAR", doc="Demand-side H2D energy flow sinks", unit="kW")
        sc.collector("C_TOT_", doc="Total costs", unit="k€/a")
        sc.collector("C_TOT_op_", doc="Total operating costs", unit="k€/a")
        sc.collector("CE_TOT_", doc="Total carbon emissions", unit="kgCO2eq/a")
        sc.collector("X_TOT_penalty_", doc="Penalty term for objective function", unit="Any")

        if sc.consider_invest:
            sc.collector("C_TOT_RMI_", doc="Total annual maintenance cost", unit="k€/a")
            sc.collector("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.collector("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")

        sc.var("C_TOT_", doc="Total costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("C_TOT_op_", doc="Total operating costs", unit="k€/a", lb=-GRB.INFINITY)
        sc.var("CE_TOT_", doc="Total emissions", unit="kgCO2eq/a", lb=-GRB.INFINITY)

        if sc.consider_invest:
            sc.param("k__r_", data=cf.main.interest_rate, doc="Calculatory interest rate")
            sc.var("C_TOT_inv_", doc="Total investment costs", unit="k€")
            sc.var("C_TOT_invAnn_", doc="Total annualized investment costs", unit="k€")
            sc.var("C_TOT_RMI_", doc="Total annual maintenance cost", unit="k€")

        sc.param("k_PTO_alpha_", data=0, doc="Pareto weighting factor")
        sc.param("k_PTO_C_", data=1, doc="Normalization factor")
        sc.param("k_PTO_CE_", data=1 / 1e4, doc="Normalization factor")
        sc.param("y_ref_", data=cf.projections.default_year, doc="Default year")
        sc.param("k_HDS_min_fac_", data=cf.main.reserve.wheight_factor, doc="Factor on cutting minimum values")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.setObjective(
            (
                (1 - p.k_PTO_alpha_) * v.C_TOT_ * p.k_PTO_C_
                + p.k_PTO_alpha_ * v.CE_TOT_ * p.k_PTO_CE_
                + quicksum(c.X_TOT_penalty_.values()) 
                + quicksum(v.S_HDS_min_TY[t,y] * p.k_HDS_min_fac_ for t in d.T for y in d.Y)
            ),
            GRB.MINIMIZE,
        )

        if sc.consider_invest:
            m.addConstr(v.C_TOT_inv_ == quicksum(c.C_TOT_inv_.values()), "investment_cost")
            m.addConstr(v.C_TOT_RMI_ == quicksum(c.C_TOT_RMI_.values()), "repair_cost")
            m.addConstr(
                v.C_TOT_invAnn_ == quicksum(c.C_TOT_invAnn_.values()), "annualized_investment_cost"
            )
            c.C_TOT_op_["RMI"] = v.C_TOT_RMI_
            c.C_TOT_["inv"] = v.C_TOT_invAnn_

        m.addConstr(v.C_TOT_op_ == quicksum(c.C_TOT_op_.values()), "operating_cost_balance")
        c.C_TOT_["op"] = v.C_TOT_op_

        m.addConstr(v.C_TOT_ == quicksum(c.C_TOT_.values()), "total_cost_balance")
        m.addConstr(
            v.CE_TOT_ == p.k__PartYearComp_ * quicksum(c.CE_TOT_.values()),
            "carbon_emission_balance",
        )
        m.addConstrs(
            (
                quicksum(x(t,y) for x in c.P_EL_source_TY.values())
                == quicksum(x(t,y) for x in c.P_EL_sink_TY.values())
                for t in d.T for y in d.Y
            ),
            "electricity_balance",
        )
        if hasattr(d, "R"):
            m.addConstrs(
                (
                    quicksum(x(t, y, r) for x in c.dH_H2D_source_TYR.values())
                    == quicksum(x(t, y, r) for x in c.dH_H2D_sink_TYR.values())
                    for t in d.T
                    for r in d.R
                    for y in d.Y
                ),
                "H2D_balance",
            )
        if hasattr(d, "R") and hasattr(d, "A"):
            m.addConstrs(
                (
                    quicksum(x(t, y, a, r) for x in c.dH_DH2D_source_TYAR.values())
                    == quicksum(x(t, y, a, r) for x in c.dH_DH2D_sink_TYAR.values())
                    for t in d.T
                    for a in d.A
                    for r in d.R
                    for y in d.Y
                ),
                "DH2D_balance",
            )
        if hasattr(d, "H"):
            m.addConstrs(
                (
                    quicksum(x(t, y, h) for x in c.dQ_heating_source_TYH.values())
                    == quicksum(x(t, y, h) for x in c.dQ_heating_sink_TYH.values())
                    for t in d.T
                    for h in d.H
                    for y in d.Y
                ),
                "heat_balance",
            )
        if hasattr(d, "N"):
            m.addConstrs(
                (
                    quicksum(x(t, y, n) for x in c.dQ_cooling_source_TYN.values())
                    == quicksum(x(t, y, n) for x in c.dQ_cooling_sink_TYN.values())
                    for t in d.T
                    for n in d.N
                    for y in d.Y
                ),
                "heat_balance",
            )


@dataclass
class EG(Component):
    """Electricity grid"""

    def param_func(self, sc: Scenario):
        sc.collector("P_EG_sell_TY", doc="Sold electricity power", unit="kW_el")
        sc.param("c_EG_buyPeak_", data=cf.comp.EG.peak_price, doc="Peak price", unit="€/kW_el/a")
        if cf.comp.EG.price_type == "fix":
            sc.param("c_EG_Y", data=cf.comp.EG.consum_price, doc="Electricity tariff",unit="€/kWh_el")
        if cf.comp.EG.price_type == "var":
            sc.param("c_EG_T", data=prep.get_var_Price_profile(), doc="Price profile EG 2019", unit="€/kWh")
        sc.param("c_EG_active_Y", data=cf.comp.EG.grid_active, doc="Electricity grid usable",unit="")
        sc.param("c_EG_addon_", data=cf.comp.EG.price_addon)
        sc.param("c_max_buy_", data=cf.comp.EG.max_buy)

        def get_cef_kw(config):
            if config == "DRAF-default":
                return dict(data=sc.prep.ce_EG_T())
            elif isinstance(config, str):
                return dict(data=hp.read(config))
            elif isinstance(config, float):
                return dict(fill=config)

        sc.param(
            "ce_EG_T",
            doc="Carbon emission factors",
            unit="kgCO2eq/kWh_el",
            **get_cef_kw(cf.comp.EG.CEFs),
        )
        sc.var("P_EG_buy_T", doc="Purchased electrical power", unit="kW_el", ub=cf.comp.EG.max_buy) # These are 
        sc.var("P_EG_sell_T", doc="Selling electrical power", unit="kW_el", ub=cf.comp.EG.max_sell) # needed for 
        sc.var("P_EG_buyPeak_", doc="Peak electrical power", unit="kW_el")                          # flexibility observation
        sc.var("P_EG_buy_TY", doc="Purchased electrical power", unit="kW_el", ub=cf.comp.EG.max_buy)
        sc.var("P_EG_sell_TY", doc="Selling electrical power", unit="kW_el", ub=cf.comp.EG.max_sell)
        sc.var("P_EG_buyPeak_Y", doc="Peak electrical power", unit="kW_el")

        sc.param("c_EG_sell_", data=cf.comp.EG.feedin_price, doc="Selected electricity tariff sell", unit="€/kWh_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(v.P_EG_buy_T[t] == quicksum(v.P_EG_buy_TY[t,y] for y in d.Y) for t in d.T)    # Here they are
        m.addConstrs(v.P_EG_sell_T[t] == quicksum(v.P_EG_sell_TY[t,y] for y in d.Y) for t in d.T)  #
        m.addConstr(v.P_EG_buyPeak_ == quicksum(v.P_EG_buyPeak_Y[y] for y in d.Y))                 #
        m.addConstrs(
            (v.P_EG_sell_TY[t,y] == quicksum(x(t,y) for x in c.P_EG_sell_TY.values()) for t in d.T for y in d.Y), "EG_sell"
        )
        m.addConstrs((v.P_EG_buy_TY[t,y] <= v.P_EG_buyPeak_Y[y] for t in d.T for y in d.Y), "EG_peak_price")
        c.P_EL_source_TY["EG"] = lambda t, y: v.P_EG_buy_TY[t,y]
        c.P_EL_sink_TY["EG"] = lambda t, y: v.P_EG_sell_TY[t,y]

        c.C_TOT_op_["EG_peak"] = quicksum(((v.P_EG_buyPeak_Y[y] * p.c_EG_buyPeak_ * conv("€", "k€", 1e-3))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        
        m.addConstrs(v.P_EG_buy_TY[t,y] <=  p.c_max_buy_ * p.c_EG_active_Y[y] for t in d.T for y in d.Y) 

        if cf.comp.EG.price_type == "fix":
            c.C_TOT_op_["EG"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum((
                    (v.P_EG_buy_TY[t,y] * (p.c_EG_Y[y] + p.c_EG_addon_) - v.P_EG_sell_TY[t,y] * p.c_EG_sell_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
                    for t in d.T for y in d.Y
                )
                * conv("€", "k€", 1e-3)
            )
        if cf.comp.EG.price_type == "var":
            c.C_TOT_op_["EG"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum((
                    (v.P_EG_buy_TY[t,y] * (p.c_EG_T[t] + p.c_EG_addon_) - (v.P_EG_sell_TY[t,y] * p.c_EG_T[t] * p.c_EG_active_Y[y]))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
                    for t in d.T for y in d.Y
                )
                * conv("€", "k€", 1e-3)
            )
        if cf.comp.EG.feedin_reduces_emissions:
            c.CE_TOT_["EG"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum(p.ce_EG_T[t] * (v.P_EG_buy_TY[t,y] - v.P_EG_sell_TY[t,y]) for t in d.T for y in d.Y)
            )
        else:
            c.CE_TOT_["EG"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum(p.ce_EG_T[t] * (v.P_EG_buy_TY[t,y]) for t in d.T for y in d.Y)
            )

    def postprocess_func(self, r: Results):
        r.make_pos_ent("P_EG_buy_TY")


@dataclass
class PPA(Component):
    """PPA Contracts EnBW"""

    def param_func(self, sc: Scenario):
        sc.param("p_PV_profile_T", data=prep.get_PV_PPA_profile(), doc="PPA profile of PV", unit="kW_el")
        sc.param("p_WTon_profile_T", data=prep.get_WTon_PPA_profile(), doc="PPA profile of OnShore Wind", unit="kW_el")
        sc.param("p_WToff_profile_T", data=prep.get_WToff_PPA_profile(), doc="PPA profile of OffShore Wind", unit="kW_el")
        sc.param("c_PV_el_", data=cf.comp.EG_PPA.PV.price, doc="PPA costs per kWh", unit="€/kWh_el")
        sc.param("c_WTon_el_", data=cf.comp.EG_PPA.OnWind.price, doc="PPA costs per kWh", unit="€/kWh_el")
        sc.param("c_WToff_el_", data=cf.comp.EG_PPA.OffWind.price, doc="PPA costs per kWh", unit="€/kWh_el")
        sc.var("P_PV_buy_cap_Y", doc="Bought PPA capacity of PV", unit="kW_el")
        sc.var("P_WTon_buy_cap_Y", doc="Bought PPA capacity of OnShore Wind", unit="kW_el")
        sc.var("P_WToff_buy_cap_Y", doc="Bought PPA capacity of OffShore Wind", unit="kW_el")
        sc.var("P_PV_PPA_FI_TY", doc="Bought PPA capacity of PV", unit="kW_el")
        sc.var("P_WTon_PPA_FI_TY", doc="Bought PPA capacity of OnShore Wind", unit="kW_el")
        sc.var("P_WToff_PPA_FI_TY", doc="Bought PPA capacity of OffShore Wind", unit="kW_el")
        sc.var("P_PV_PPA_OC_TY", doc="Bought PPA capacity of PV", unit="kW_el")
        sc.var("P_WTon_PPA_OC_TY", doc="Bought PPA capacity of OnShore Wind", unit="kW_el")
        sc.var("P_WToff_PPA_OC_TY", doc="Bought PPA capacity of OffShore Wind", unit="kW_el")
    
    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):

        if cf.comp.EG.price_type == "fix":
            m.addConstrs(v.P_PV_buy_cap_Y[y] * p.p_PV_profile_T[t] == v.P_PV_PPA_FI_TY[t,y] + v.P_PV_PPA_OC_TY[t,y] for t in d.T for y in d.Y)
            m.addConstrs(v.P_WTon_buy_cap_Y[y] * p.p_WTon_profile_T[t] == v.P_WTon_PPA_FI_TY[t,y] + v.P_WTon_PPA_OC_TY[t,y] for t in d.T for y in d.Y)
            m.addConstrs(v.P_WToff_buy_cap_Y[y] * p.p_WToff_profile_T[t] == v.P_WToff_PPA_FI_TY[t,y] + v.P_WToff_PPA_OC_TY[t,y] for t in d.T for y in d.Y)

        if cf.comp.EG.price_type == "var":
            m.addConstrs(v.P_PV_buy_cap_Y[y] * p.p_PV_profile_T[t] * (1-p.c_EG_active_Y[y]) == v.P_PV_PPA_FI_TY[t,y] + v.P_PV_PPA_OC_TY[t,y] for t in d.T for y in d.Y)
            m.addConstrs(v.P_WTon_buy_cap_Y[y] * p.p_WTon_profile_T[t] * (1-p.c_EG_active_Y[y]) == v.P_WTon_PPA_FI_TY[t,y] + v.P_WTon_PPA_OC_TY[t,y] for t in d.T for y in d.Y)
            m.addConstrs(v.P_WToff_buy_cap_Y[y] * p.p_WToff_profile_T[t] * (1-p.c_EG_active_Y[y]) == v.P_WToff_PPA_FI_TY[t,y] + v.P_WToff_PPA_OC_TY[t,y] for t in d.T for y in d.Y)

        c.P_EL_source_TY["PV_PPA"] = lambda t,y: v.P_PV_PPA_FI_TY[t,y] + v.P_PV_PPA_OC_TY[t,y]
        c.P_EL_source_TY["WTon_PPA"] = lambda t,y: v.P_WTon_PPA_FI_TY[t,y] + v.P_WTon_PPA_OC_TY[t,y]
        c.P_EL_source_TY["WToff_PPA"] = lambda t,y: v.P_WToff_PPA_FI_TY[t,y] + v.P_WToff_PPA_OC_TY[t,y]
        c.P_EG_sell_TY["PV_PPA"] = lambda t,y: v.P_PV_PPA_FI_TY[t,y]
        c.P_EG_sell_TY["WTon_PPA"] = lambda t,y: v.P_WTon_PPA_FI_TY[t,y]
        c.P_EG_sell_TY["WToff_PPA"] = lambda t,y: v.P_WToff_PPA_FI_TY[t,y] 
        
        c.C_TOT_op_["PV_PPA"] = (
            quicksum(((p.c_PV_el_
            * conv("€", "k€", 1e-3)
            * quicksum(v.P_PV_buy_cap_Y[y] * p.p_PV_profile_T[t] for t in d.T)
            * p.k__PartYearComp_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
            for y in d.Y
            )
        )
        c.C_TOT_op_["WTon_PPA"] = (
            quicksum(((p.c_WTon_el_
            * conv("€", "k€", 1e-3)
            * quicksum(v.P_WTon_buy_cap_Y[y] * p.p_WTon_profile_T[t] for t in d.T)
            * p.k__PartYearComp_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
            for y in d.Y
            )
        )
        c.C_TOT_op_["WToff_PPA"] = (
            quicksum(((p.c_WToff_el_
            * conv("€", "k€", 1e-3)
            * quicksum(v.P_WToff_buy_cap_Y[y] * p.p_WToff_profile_T[t] for t in d.T)
            * p.k__PartYearComp_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
            for y in d.Y
            )
        )

@dataclass
class BES(Component):
    """Battery Energy Storage"""

    def param_func(self, sc: Scenario):
        sc.param(
            "E_BES_CAPx_", data=cf.comp.BES.existing_capa, doc="Existing capacity", unit="kWh_el"
        )
        sc.param(
            "k_BES_ini_",
            data=cf.comp.BES.initial_and_final_filling_level,
            doc="Initial and final energy filling share",
        )
        sc.param("eta_BES_ch_", data=cf.comp.BES.eta_cycle**0.5, doc="Charging efficiency")
        sc.param("eta_BES_dis_", data=cf.comp.BES.eta_cycle**0.5, doc="Discharging efficiency")
        sc.param("eta_BES_self_", data=cf.comp.BES.eta_self, doc="Self-discharge rate")
        sc.param(
            "k_BES_inPerCap_", cf.comp.BES.in_per_capa, doc="Maximum charging power per capacity"
        )
        sc.param(
            "k_BES_outPerCap_",
            cf.comp.BES.out_per_capa,
            doc="Maximum discharging power per capacity",
        )
        sc.var("E_BES_TY", doc="Electricity stored", unit="kWh_el")
        sc.var("P_BES_in_TY", doc="Charging power", unit="kW_el")
        sc.var("P_BES_out_TY", doc="Discharging power", unit="kW_el")

        if sc.consider_invest:
            sc.param("k_BES_RMI_", data=cf.comp.BES.RMI, doc=Descs.RMI.en)
            sc.param("N_BES_", data=cf.comp.BES.OL, doc=Etypes.N.en, unit="a")
            sc.param("z_BES_", data=1, doc="If new capacity is allowed")
            sc.param("c_BES_inv_", data=cf.comp.BES.CapEx, doc="Cost per kWh", unit="€/kWh")
            sc.var("E_BES_CAPn_Y", doc="New capacity", unit="kWh_el")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        """Note: In this model does not prevent simultaneous charging and discharging,
        which can appear with negative electricity prices. To avoid this behaviour expensive binary
        variables can be introduced, e.g., like in
        AmirMansouri.2021: https://doi.org/10.1016/j.seta.2021.101376
        """

        m.addConstrs(
            (v.P_BES_in_TY[t,y] <= p.k_BES_inPerCap_ * (p.E_BES_CAPx_ + v.E_BES_CAPn_Y[y]) for t in d.T for y in d.Y), "BES_limit_charging_power"
        )
        m.addConstrs(
            (v.P_BES_out_TY[t,y] <= p.k_BES_outPerCap_ * (p.E_BES_CAPx_ + v.E_BES_CAPn_Y[y]) for t in d.T for y in d.Y),
            "BES_limit_discharging_power",
        )
        m.addConstrs((v.E_BES_TY[t,y] <= (p.E_BES_CAPx_ + v.E_BES_CAPn_Y[y]) for t in d.T for y in d.Y), "BES_limit_cap")
        m.addConstr((v.E_BES_TY[d.T[-1],d.Y[-1]] >= p.k_BES_ini_ * (p.E_BES_CAPx_ + v.E_BES_CAPn_Y[d.Y[0]])), "BES_last_timestep")
        m.addConstrs(
            (
                v.E_BES_TY[t,y]
                == (p.k_BES_ini_ * (p.E_BES_CAPx_ + v.E_BES_CAPn_Y[d.Y[0]]) if (t == d.T[0] and y == d.Y[0]) else (v.E_BES_TY[d.T[-1],d.Y[d.Y.index(y)-1]] if t == d.T[0] else v.E_BES_TY[t - 1,y]))
                * (1 - p.eta_BES_self_ * p.k__dT_)
                + (v.P_BES_in_TY[t,y] * p.eta_BES_ch_ - v.P_BES_out_TY[t,y] / p.eta_BES_dis_) * p.k__dT_
                for t in d.T for y in d.Y
            ),
            "BES_electricity_balance",
        )
        c.P_EL_source_TY["BES"] = lambda t,y: v.P_BES_out_TY[t,y]
        c.P_EL_sink_TY["BES"] = lambda t,y: v.P_BES_in_TY[t,y]
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.E_BES_CAPn_Y[y] >= v.E_BES_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs((v.E_BES_CAPn_Y[y] <= p.z_BES_ * 1e6 for y in d.Y), "BES_limit_new_capa")
            c.C_TOT_inv_["BES"] = v.E_BES_CAPn_Y[d.Y[-1]] * p.c_BES_inv_ * conv("€", "k€", 1e-3)
            c.C_TOT_invAnn_["BES"] = quicksum((((v.E_BES_CAPn_Y[y] * p.c_BES_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_BES_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["BES"] = quicksum((((v.E_BES_CAPn_Y[y] * p.c_BES_inv_ * conv("€", "k€", 1e-3)) * p.k_BES_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)


@dataclass
class OnPV(Component):
    """Onsite-Photovoltaic System

    No network charges since located at demand side.
    """

    def param_func(self, sc: Scenario):
        sc.param("P_OnPV_CAPx_", data=cf.comp.OnPV.P_exist, doc="Existing capacity", unit="kW_peak")
        sc.prep.P_PV_profile_T("P_OnPV_profile_T", use_coords=True, tilt=cf.comp.OnPV.tilt) # Let's keep it that way, we don't know the weather yet
        # ensure_capacity factor:
        sc.params.P_OnPV_profile_T *= (
            cf.comp.OnPV.capacity_factor / sc.params.P_OnPV_profile_T.mean()
        )
        sc.var("P_OnPV_FI_TY", doc="Feed-in", unit="kW_el")
        sc.var("P_OnPV_OC_TY", doc="Own consumption", unit="kW_el")
        sc.param(
            "k_OnPV_AreaPerPeak_",
            data=cf.comp.OnPV.specific_area,
            doc="Area efficiency of new PV",
            unit="m²/kW_peak",
        )
        sc.param(
            "A_OnPV_avail_",
            data=cf.comp.OnPV.A_available,
            doc="Area available for new PV",
            unit="m²",
        )

        if sc.consider_invest:
            sc.param("z_OnPV_", data=1, doc="If new capacity is allowed")
            sc.param("c_OnPV_inv_", data=cf.comp.OnPV.CapEx, unit="€/kW_peak")
            sc.param("k_OnPV_RMI_", data=cf.comp.OnPV.RMI, unit="€/kW_peak")
            sc.param("N_OnPV_", data=cf.comp.OnPV.OL, unit="a")
            sc.var("P_OnPV_CAPn_Y", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            ((p.P_OnPV_CAPx_ + v.P_OnPV_CAPn_Y[y]) * p.P_OnPV_profile_T[t] == v.P_OnPV_FI_TY[t,y] + v.P_OnPV_OC_TY[t,y] for t in d.T for y in d.Y),
            "OnPV_balance",
        )
        c.P_EL_source_TY["OnPV"] = lambda t,y: v.P_OnPV_FI_TY[t,y] + v.P_OnPV_OC_TY[t,y]
        c.P_EG_sell_TY["OnPV"] = lambda t,y: v.P_OnPV_FI_TY[t,y]
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.P_OnPV_CAPn_Y[y] >= v.P_OnPV_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs(
                (v.P_OnPV_CAPn_Y[y] <= p.z_OnPV_ * p.A_OnPV_avail_ / p.k_OnPV_AreaPerPeak_ for y in d.Y),
                "OnPV_limit_capn",
            )
            c.C_TOT_inv_["OnPV"] = (v.P_OnPV_CAPn_Y[d.Y[-1]] * p.c_OnPV_inv_ * conv("€", "k€", 1e-3))
            c.C_TOT_invAnn_["OnPV"] = quicksum((((v.P_OnPV_CAPn_Y[y] * p.c_OnPV_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_OnPV_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["OnPV"] = quicksum((((v.P_OnPV_CAPn_Y[y] * p.c_OnPV_inv_ * conv("€", "k€", 1e-3)) * p.k_OnPV_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)


@dataclass
class PV(Component):
    """Offsite-Photovoltaic System

    Network charges are paid for own-consumed energy, since not located at demand side.
    """

    def param_func(self, sc: Scenario):
        sc.param("P_PV_CAPx_", data=cf.comp.PV.P_exist, doc="Existing capacity", unit="kW_peak")
        sc.prep.P_PV_profile_T(use_coords=True, tilt=cf.comp.PV.tilt)
        # ensure_capacity factor:
        sc.params.P_PV_profile_T *= cf.comp.PV.capacity_factor / sc.params.P_PV_profile_T.mean()
        sc.var("P_PV_FI_TY", doc="Feed-in", unit="kW_el")
        sc.var("P_PV_OC_TY", doc="Own consumption", unit="kW_el")
        sc.param("P_PV_max_", data=cf.comp.PV.P_max, doc="Maximum new PV", unit="kW_peak")

        if sc.consider_invest:
            sc.param("z_PV_", data=1, doc="If new capacity is allowed")
            sc.param("c_PV_inv_", data=cf.comp.PV.CapEx, unit="€/kW_peak")
            sc.param("k_PV_RMI_", data=cf.comp.PV.RMI, unit="€/kW_peak")
            sc.param("N_PV_", data=cf.comp.PV.OL, unit="a")
            sc.var("P_PV_CAPn_Y", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            ((p.P_PV_CAPx_ + v.P_PV_CAPn_Y[y]) * p.P_PV_profile_T[t] == v.P_PV_FI_TY[t,y] + v.P_PV_OC_TY[t,y] for t in d.T for y in d.Y),
            "PV_balance",
        )
        c.P_EL_source_TY["PV"] = lambda t,y: v.P_PV_FI_TY[t,y] + v.P_PV_OC_TY[t,y]
        c.P_EG_sell_TY["PV"] = lambda t,y: v.P_PV_FI_TY[t,y]
        c.C_TOT_op_["PV"] = (
            quicksum(((p.k__dT_
            * p.k__PartYearComp_
            * quicksum(v.P_PV_OC_TY[t,y] for t in d.T)
            * p.c_EG_addon_
            * conv("€", "k€", 1e-3))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
            for y in d.Y
            )
        )
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.P_PV_CAPn_Y[y] >= v.P_PV_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs((v.P_PV_CAPn_Y[y] <= p.z_PV_ * p.P_PV_max_ for y in d.Y), "PV_limit_capn")
            c.C_TOT_inv_["PV"] = (v.P_PV_CAPn_Y[d.Y[-1]] * p.c_PV_inv_ * conv("€", "k€", 1e-3))
            c.C_TOT_invAnn_["PV"] = quicksum((((v.P_PV_CAPn_Y[y] * p.c_PV_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_PV_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_)**(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["PV"] = quicksum((((v.P_PV_CAPn_Y[y] * p.c_PV_inv_ * conv("€", "k€", 1e-3)) * p.k_PV_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_)**(y - p.y_ref_)) for y in d.Y)


@dataclass
class WT(Component):
    """Wind turbine

    Args:
        pay_network_tariffs: If network charges are paid for own-consumed energy (offsite).
    """

    pay_network_tariffs: bool = True

    def param_func(self, sc: Scenario):
        sc.param("P_WT_CAPx_", data=0, doc="Existing capacity", unit="kW_peak")
        sc.param(
            "P_WT_profile_T",
            data=prep.get_WTon_Ka_profile(),
            doc="Wind profile",
            unit="kW_el",
        ) # Let's keep it that way, we don't know the weather yet
        sc.param(
            "y_WT_pnt_",
            data=int(self.pay_network_tariffs),
            doc="If `c_EG_addon_` is paid on own wind energy consumption (e.g. for off-site PPA)",
        )
        sc.var("P_WT_FI_TY", doc="Feed-in", unit="kW_el")
        sc.var("P_WT_OC_TY", doc="Own consumption", unit="kW_el")

        if sc.consider_invest:
            sc.param(
                "P_WT_max_", data=cf.comp.WT.P_max, doc="Maximum installed capacity", unit="kW_peak"
            )
            sc.param("z_WT_", data=1, doc="If new capacity is allowed")
            sc.param(
                "c_WT_inv_",
                data=cf.comp.WT.CapEx,
                doc="CAPEX",
                unit="€/kW_peak",
                src="https://windeurope.org/newsroom/press-releases/europe-invested-41-bn-euros-in-new-wind-farms-in-2021",
            )  # or 1118.77 €/kWp invest and 27 years operation life for onshore wind https://github.com/PyPSA/technology-data/blob/4eaddec90f429246445f08476b724393dde753c8/outputs/costs_2020.csv
            sc.param(
                "k_WT_RMI_",
                data=cf.comp.WT.RMI,
                doc="RMI",
                unit="",
                src="https://www.npro.energy/main/en/help/economic-parameters",
            )
            sc.param(
                "N_WT_",
                data=cf.comp.WT.OL,
                doc="Operation life",
                unit="a",
                src="https://www.twi-global.com/technical-knowledge/faqs/how-long-do-wind-turbines-last",
            )
            sc.var("P_WT_CAPn_Y", doc="New capacity", unit="kW_peak")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            ((p.P_WT_CAPx_ + v.P_WT_CAPn_Y[y]) * p.P_WT_profile_T[t] == v.P_WT_FI_TY[t,y] + v.P_WT_OC_TY[t,y] for t in d.T for y in d.Y),
            "WT_balance",
        )
        c.P_EL_source_TY["WT"] = lambda t,y: v.P_WT_FI_TY[t,y] + v.P_WT_OC_TY[t,y]
        c.P_EG_sell_TY["WT"] = lambda t,y: v.P_WT_FI_TY[t,y]

        if p.y_WT_pnt_:
            c.C_TOT_op_["WT"] = (
                quicksum(((p.k__dT_
                * p.k__PartYearComp_
                * quicksum(v.P_WT_OC_TY[t,y] for t in d.T)
                * p.c_EG_addon_
                * conv("€", "k€", 1e-3))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
                for y in d.Y
                )
            )
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.P_WT_CAPn_Y[y] >= v.P_WT_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs((v.P_WT_CAPn_Y[y] <= p.z_WT_ * p.P_WT_max_ for y in d.Y), "WT_limit_capn")
            c.C_TOT_inv_["WT"] = (v.P_WT_CAPn_Y[d.Y[-1]] * p.c_WT_inv_ * conv("€", "k€", 1e-3))
            c.C_TOT_invAnn_["WT"] = quicksum((((v.P_WT_CAPn_Y[y] * p.c_WT_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_WT_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["WT"] = quicksum((((v.P_WT_CAPn_Y[y] * p.c_WT_inv_ * conv("€", "k€", 1e-3)) * p.k_WT_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)


@dataclass
class EHP(Component):
    """Electric heat pump"""

    time_dependent_amb: bool = True
    n: int = 1

    def dim_func(self, sc: Scenario):
        sc.dim("N", data=cf.main.cool_temp_levels, doc="Excess heat temperature levels")
        sc.dim("H", data=cf.main.heat_temp_levels, doc="Heating temperature levels")
        sc.param("T_heat_T", data=prep.get_DHN_curve(), doc="Temperature of heat level", unit="°C")
        sc.param("T_N", data=cf.main.cool_temperatures, doc="Temperature of cool level", unit="°C")
        sc.dim("E", data=cf.comp.EHP.cooling_levels, doc="Evaporation temperature levels")
        sc.dim("C", data=cf.comp.EHP.heating_levels, doc="Condensing temperature levels")

    def param_func(self, sc: Scenario):
        p = sc.params

        if self.time_dependent_amb:
            sc.prep.T__amb_T()
        sc.param("T__amb_", data=25, doc="Approximator for ambient air", unit="°C")

        sc.param(
            "T_EHP_rhine_T", data=prep.T_EHP_rhine_T(sc), doc="Rhine water temperature", unit="°C" # Let's keep it that way, we don' know the weather yet
        )

        sc.param(
            "T_EHP_Cond_T",
            data=p.T_heat_T,
            doc="Condensation side temperature",
            unit="°C",
        )
        sc.param("T_EHP_Eva_E", data=p.T_N - 5, doc="Evaporation side temperature", unit="°C")
        sc.param("n_EHP_", data=self.n, doc="Maximum number of parallel operation modes")
        sc.param(
            "eta_EHP_",
            data=0.5,
            doc="Ratio of reaching the ideal COP (exergy efficiency)",
            src="@Arat_2017",
            # Cox_2022 used 0.45: https://doi.org/10.1016/j.apenergy.2021.118499
            # but roughly 0.5 in recent real operation of high temperature EHP: https://www.waermepumpe.de/fileadmin/user_upload/waermepumpe/01_Verband/Webinare/Vortrag_Wilk_AIT_02062020.pdf
        )
        sc.param("dQ_EHP_CAPx_", data=0, doc="Existing heating capacity", unit="kW_th")
        sc.param(
            "dQ_EHP_max_", data=1e5, doc="Big-M number (upper bound for CAPn + CAPx)", unit="kW_th"
        )
        sc.var("P_EHP_TYEC", doc="Consuming power", unit="kW_el")
        sc.var("dQ_EHP_Cond_TYEC", doc="Heat flow released on condensation side", unit="kW_th")
        sc.var("dQ_EHP_Eva_TYEC", doc="Heat flow absorbed on evaporation side", unit="kW_th")
        sc.var("Y_EHP_TYEC", doc="If source and sink are connected at time-step", vtype=GRB.BINARY)

        if sc.consider_invest:
            sc.param("z_EHP_", data=1, doc="If new capacity is allowed")
            sc.param("k_EHP_RMI_", data=cf.comp.EHP.RMI, doc=Descs.RMI.en)
            sc.param("N_EHP_", data=cf.comp.EHP.OL, doc=Etypes.N.en, unit="a")
            sc.param("c_EHP_inv_", data=cf.comp.EHP.CapEx, doc="CapEx", unit="€/kW_th")
            sc.var("dQ_EHP_CAPn_Y", doc="New heating capacity", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        def get_cop(t, e):
            T_amb = p.T__amb_T[t]
            T_rhine = p.T_EHP_rhine_T[t]
            T_cond = p.T_EHP_Cond_T[t]
            T_eva = T_rhine if e == "rhine_water" else (T_amb if e == "amb" else p.T_EHP_Eva_E[e])
            return 100 if T_cond <= T_eva else p.eta_EHP_ * (T_cond + 273) / (T_cond - T_eva)

        m.addConstrs(
            (
                v.dQ_EHP_Cond_TYEC[t, y, e, c] == v.P_EHP_TYEC[t, y, e, c] * get_cop(t, e)
                for t in d.T
                for e in d.E
                for c in d.C
                for y in d.Y
            ),
            "HP_balance_1",
        )
        m.addConstrs(
            (
                v.dQ_EHP_Cond_TYEC[t, y, e, c] == v.dQ_EHP_Eva_TYEC[t, y, e, c] + v.P_EHP_TYEC[t, y, e, c]
                for t in d.T
                for e in d.E
                for c in d.C
                for y in d.Y
            ),
            "HP_balance_2",
        )
        m.addConstrs(
            (
                v.dQ_EHP_Cond_TYEC[t, y, e, c] <= v.Y_EHP_TYEC[t, y, e, c] * p.dQ_EHP_max_
                for t in d.T
                for e in d.E
                for c in d.C
                for y in d.Y
            ),
            "HP_bigM",
        )
        if ("amb" in d.E) and ("amb" in d.C):
            # Avoid ambient-to-ambient EHP operation occur due to negative electricity prices
            m.addConstr((v.Y_EHP_TYEC.sum("*", "*", "amb", "amb") == 0), "HP_no_amb_to_amb")
        m.addConstrs((v.dQ_EHP_Cond_TYEC.sum(t, y, "*", "*") <= (p.dQ_EHP_CAPx_ + v.dQ_EHP_CAPn_Y[y]) for t in d.T for y in d.Y), "HP_limit_cap")
        m.addConstrs((v.Y_EHP_TYEC.sum(t, y, "*", "*") <= p.n_EHP_ for t in d.T for y in d.Y), "HP_operating_mode")

        c.P_EL_sink_TY["EHP"] = lambda t,y: v.P_EHP_TYEC.sum(t, y, "*", "*")
        c.dQ_cooling_sink_TYN["EHP"] = lambda t, y, n: v.dQ_EHP_Eva_TYEC.sum(t, y, n, "*")
        c.dQ_heating_source_TYH["EHP"] = lambda t, y, h: v.dQ_EHP_Cond_TYEC.sum(t, y, "*", h)
        c.dQ_amb_sink_TY["EHP"] = lambda t, y: v.dQ_EHP_Eva_TYEC.sum(t, "*","amb", "*") 
        c.dQ_rhine_sink_TY["EHP"] = lambda t, y: v.dQ_EHP_Eva_TYEC.sum(t, "*","rhine_water", "*") 
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dQ_EHP_CAPn_Y[y] >= v.dQ_EHP_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs((v.dQ_EHP_CAPn_Y[y] <= p.z_EHP_ * 1e6 for y in d.Y), "HP_limit_capn")
            c.C_TOT_inv_["EHP"] = (v.dQ_EHP_CAPn_Y[d.Y[-1]] * p.c_EHP_inv_ * conv("€", "k€", 1e-3))
            c.C_TOT_invAnn_["EHP"] = quicksum((((v.dQ_EHP_CAPn_Y[y] * p.c_EHP_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_EHP_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["EHP"] = quicksum((((v.dQ_EHP_CAPn_Y[y] * p.c_EHP_inv_ * conv("€", "k€", 1e-3)) * p.k_EHP_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)


@dataclass
class TES(Component):
    """Thermal energy storage"""

    def dim_func(self, sc: Scenario):
        sc.dim(
            "L",
            data=cf.comp.TES.temp_levels,
            doc="Thermal demand temperature levels (inlet / outlet) in °C",
        )

    def param_func(self, sc: Scenario):
        sc.param("eta_TES_self_", data=cf.comp.TES.eta_self, doc="Self-discharge rate")
        sc.param(
            "k_TES_inPerCap_", data=cf.comp.TES.in_per_capa, doc="Ratio loading power / capacity"
        )
        sc.param(
            "k_TES_outPerCap_", data=cf.comp.TES.out_per_capa, doc="Ratio loading power / capacity"
        )
        sc.param(
            "k_TES_ini_L",
            fill=cf.comp.TES.initial_and_final_filling_level,
            doc="Initial and final energy level share",
        )
        sc.var("dQ_TES_in_TYL", doc="Storage input heat flow", unit="kW_th")
        sc.var("dQ_TES_out_TYL", doc="Storage input heat flow", unit="kW_th")
        sc.var("Q_TES_TYL", doc="Stored heat", unit="kWh_th")

        if sc.consider_invest:
            sc.param("z_TES_L", fill=1, doc="If new capacity is allowed")
            sc.param("c_TES_inv_", data=cf.comp.TES.CapEx, doc="CapEx", unit="€/kWh_th")
            sc.param("k_TES_RMI_", data=cf.comp.TES.RMI, doc=Descs.RMI.en)
            sc.param("N_TES_", data=cf.comp.TES.OL, doc=Etypes.N.en, unit="a")
            sc.var("Q_TES_CAPn_LY", doc="New capacity", unit="kWh_th", ub=1e7)

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.Q_TES_TYL[t, y, l]
                == ((p.k_TES_ini_L[l] * v.Q_TES_CAPn_LY[l,d.Y[0]]) if (t == d.T[0] and y == d.Y[0]) else (v.Q_TES_TYL[d.T[-1], d.Y[d.Y.index(y)-1], l] if t == d.T[0] else v.Q_TES_TYL[t - 1, y, l]))
                * (1 - p.eta_TES_self_ * p.k__dT_)
                + p.k__dT_ * v.dQ_TES_in_TYL[t, y, l]
                - p.k__dT_ * v.dQ_TES_out_TYL[t, y, l]
                for t in d.T
                for l in d.L
                for y in d.Y
            ),
            "TES_balance",
        )
        m.addConstrs(
            (v.Q_TES_TYL[t, y, l] <= v.Q_TES_CAPn_LY[l,y] for t in d.T for l in d.L for y in d.Y), "TES_limit_cap"
        )
        m.addConstrs(
            (
                v.dQ_TES_in_TYL[t, y, l] <= p.k_TES_inPerCap_ * v.Q_TES_CAPn_LY[l,y]
                for t in d.T
                for l in d.L
                for y in d.Y
            ),
            "TES_limit_in",
        )
        m.addConstrs(
            (
                v.dQ_TES_out_TYL[t, y, l] <= p.k_TES_outPerCap_ * v.Q_TES_CAPn_LY[l,y]
                for t in d.T
                for l in d.L
                for y in d.Y
            ),
            "TES_limit_out",
        )
        m.addConstrs(
            (v.Q_TES_TYL[d.T[-1], d.Y[-1], l] >= p.k_TES_ini_L[l] * v.Q_TES_CAPn_LY[l,d.Y[0]] for l in d.L),
            "TES_last_timestep",
        )

        # only sink here, since dQ_TES_in_TL is also defined for negative
        # values to reduce number of variables:
        c.dQ_cooling_sink_TYN["TES"] = lambda t, y, n: v.dQ_TES_in_TYL[t, y, n] if n in d.L else 0
        c.dQ_cooling_source_TYN["TES"] = lambda t, y, n: v.dQ_TES_out_TYL[t, y, n] if n in d.L else 0
        #c.dQ_heating_sink_TYH["TES"] = lambda t, y, h: v.dQ_TES_in_TYL[t, y, h] if h in d.L else 0
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.Q_TES_CAPn_LY[l,y] >= v.Q_TES_CAPn_LY[l,d.Y[d.Y.index(y)-1]] for l in d.L for y in d.Y[1:]), "Installed power can only increase")
        if sc.consider_invest:
            m.addConstrs((v.Q_TES_CAPn_LY[l,y] <= p.z_TES_L[l] * 1e5 for l in d.L for y in d.Y), "TES_limit_capn")
            c.C_TOT_inv_["TES"] = quicksum(v.Q_TES_CAPn_LY[l, d.Y[-1]] * p.c_TES_inv_ * conv("€", "k€", 1e-3) for l in d.L)
            c.C_TOT_invAnn_["TES"] = quicksum((quicksum((v.Q_TES_CAPn_LY[l,y] * p.c_TES_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_TES_) for l in d.L)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
            c.C_TOT_RMI_["TES"] = quicksum((quicksum((v.Q_TES_CAPn_LY[l,y] * p.c_TES_inv_ * conv("€", "k€", 1e-3)) * p.k_TES_RMI_ for l in d.L)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)

    def postprocess_func(self, r: Results):
        r.make_pos_ent("dQ_TES_in_TYL", "dQ_TES_out_TYL", "Storage output heat flow")


@dataclass
class Elc(Component):
    """Electrolyzer"""

    def param_func(self, sc: Scenario):
        my_elc = cf.comp.Elc.tecs[cf.comp.Elc.chosen_tec]
        sc.param("eta_Elc_H_", data=((my_elc.eta.t0+my_elc.eta.t10)/2), doc="Efficiency of hydrogen production")
        sc.param("eta_Elc_th_", data=((my_elc.eta_th.t0+my_elc.eta_th.t10)/2), doc="Efficiency of heat production")
        sc.param("N_Elc_", data=my_elc.OL, doc=Etypes.N.en, unit="a")
        sc.param("c_Elc_inv_", data=my_elc.CapEx, doc="CapEx", unit="€/kWh_el")
        sc.param("k_Elc_RMI_", data=my_elc.RMI, doc=Descs.RMI.en)
        sc.param(
            "k_Elc_water_", data=my_elc.water_rate, doc="Specific water demand", unit="kg_H2O/kg_H2"
        )
        sc.param(
            "c_Elc_water_",
            data=cf.comp.Elc.water_price,
            doc="Price of demineralized water",
            unit="€/t",
        )
        sc.var("M_Elc_H2_TY", doc="Total hydrogen production", unit="kg/h")
        sc.var("M_Elc_water_TY", doc="Total water consumption", unit="kg/h")
        sc.var("P_Elc_CAPn_Y", doc="New capacity", unit="kW_el")
        sc.var("P_Elc_TY", doc="Consumed electrical power", unit="kW_el")
        sc.var("dH_Elc_TY", doc="Hydrogen generation power", unit="kW")
        sc.var("dQ_Elc_TY", doc="Produced heat flow", unit="kW_th")
        sc.var("dQ_Elc_HP_TY", doc="Produced heat flow", unit="kW_th")
        sc.var("dQ_Elc_amb_TY", doc="Produced heat flow", unit="kW_th")
        sc.var("dQ_Elc_rhine_TY", doc="Produced heat flow", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs((v.P_Elc_TY[t,y] <= v.P_Elc_CAPn_Y[y] for t in d.T for y in d.Y), "Elc_limit")
        m.addConstrs((v.dH_Elc_TY[t,y] == v.P_Elc_TY[t,y] * p.eta_Elc_H_ for t in d.T for y in d.Y), "Elc_hydrogen")
        m.addConstrs((v.dQ_Elc_TY[t,y] == v.P_Elc_TY[t,y] * p.eta_Elc_th_ for t in d.T for y in d.Y), "Elc_thermal")
        m.addConstrs(
            (
                v.M_Elc_H2_TY[t,y]
                == v.dH_Elc_TY[t,y] * p.k__dT_ * p.k__PartYearComp_ / cf.energy_density.H2 for t in d.T for y in d.Y
            ),
            "Elc_H2 per hour",
        )
        m.addConstrs((v.M_Elc_water_TY[t,y] == v.M_Elc_H2_TY[t,y] * p.k_Elc_water_ for t in d.T for y in d.Y), "Elc_water per hour")

        m.addConstrs((v.dQ_Elc_TY[t,y] == v.dQ_Elc_HP_TY[t,y] + v.dQ_Elc_amb_TY[t,y] + v.dQ_Elc_rhine_TY[t,y] for t in d.T for y in d.Y), "Split energy as wished")

        c.P_EL_sink_TY["Elc"] = lambda t,y: v.P_Elc_TY[t,y]
        c.dH_H2D_source_TYR["Elc"] = lambda t, y, r: v.dH_Elc_TY[t,y] if r == "CH2" else 0
        c.dQ_cooling_source_TYN["Elc"] = lambda t, y, n: v.dQ_Elc_HP_TY[t,y] if n == "Elc" else 0
        c.dQ_amb_source_TY["Elc"] = lambda t, y: v.dQ_Elc_amb_TY[t,y] 
        c.dQ_rhine_source_TY["Elc"] = lambda t, y: v.dQ_Elc_rhine_TY[t,y]
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.P_Elc_CAPn_Y[y] >= v.P_Elc_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        c.C_TOT_inv_["Elc"] = (v.P_Elc_CAPn_Y[d.Y[-1]] * p.c_Elc_inv_ * conv("€", "k€", 1e-3))
        c.C_TOT_invAnn_["Elc"] = quicksum((((v.P_Elc_CAPn_Y[y] * p.c_Elc_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_Elc_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["Elc"] = quicksum((((v.P_Elc_CAPn_Y[y] * p.c_Elc_inv_ * conv("€", "k€", 1e-3)) * p.k_Elc_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)

        c.C_TOT_op_["Elc"] = (
            quicksum(((p.c_Elc_water_
            * conv("€", "k€", 1e-3)
            * conv("1/t", "1/kg", 1e-3)
            * quicksum(v.M_Elc_water_TY[t,y] for t in d.T)
            * p.k__PartYearComp_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_))
            for y in d.Y
            )
        )


@dataclass
class HDS(Component):
    """Hydrogen Derivative Storage"""

    def param_func(self, sc: Scenario):
        df = pd.DataFrame(cf.comp.HDS).T
        sc.param("c_HDS_inv_R", data=df["CapEx"], doc="Specific investment cost", unit="€/kWh")
        sc.param("k_HDS_RMI_R", data=df["RMI"], doc=Descs.RMI.en)
        sc.param("N_HDS_R", data=df["OL"], doc=Etypes.N.en, unit="a")
        sc.param("k_HDS_inPerCap_R", data=df["in_per_cap"])
        sc.param("k_HDS_outPerCap_R", data=df["out_per_cap"])
        sc.param("eta_HDS_ch_R", data=df["eta_in"])
        sc.param("eta_HDS_dis_R", data=df["eta_out"])
        sc.param("eta_HDS_self_R", data=df["eta_self"])
        sc.param("k_HDS_ini_R", data=df["k_ini"], doc="Initial and final energy filling share")
        sc.param("q_HDS_min_Y", data=cf.main.reserve.amount, doc="Minimal reserve in storage")
        sc.param("d_HDS_min_reserve_hours_", data=cf.main.reserve.equals_hours, doc="Hours of reserve")
        sc.param("d_HDS_hours_cut_", data=cf.main.reserve.time_to_reach, doc="Days until minimum reserve has to be reached")
        sc.var("H_HDS_CAPn_RY", doc="New capacity", unit="kWh")
        sc.var("H_HDS_TYR", doc="H2D energy stored", unit="kWh")
        sc.var("dH_HDS_in_TYR", doc="Input power", unit="kW")
        sc.var("dH_HDS_out_TYR", doc="Output power", unit="kW")
        sc.var("S_HDS_min_TY", doc="Slack variable for minimum power", unit="kWh")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.dH_HDS_in_TYR[t, y, r] <= p.k_HDS_inPerCap_R[r] * v.H_HDS_CAPn_RY[r,y]
                for t in d.T
                for r in d.R
                for y in d.Y
            ),
            "HDS_limit_charging_power",
        )
        m.addConstrs(
            (
                v.dH_HDS_out_TYR[t, y, r] <= p.k_HDS_outPerCap_R[r] * v.H_HDS_CAPn_RY[r,y]
                for t in d.T
                for r in d.R
                for y in d.Y
            ),
            "HDS_limit_discharging_power",
        )
        m.addConstrs(
            (v.H_HDS_TYR[t, y, r] <= v.H_HDS_CAPn_RY[r,y] for t in d.T for r in d.R for y in d.Y), "HDS_limit_cap"
        )
        m.addConstrs(
            (v.H_HDS_TYR[d.T[-1], d.Y[-1], r] >= p.k_HDS_ini_R[r] * v.H_HDS_CAPn_RY[r,d.Y[0]] for r in d.R),
            "HDS_last_timestep",
        )
        m.addConstrs(
            (
                v.H_HDS_TYR[t, y, r]
                == (
                    p.k_HDS_ini_R[r] * v.H_HDS_CAPn_RY[r,d.Y[0]]
                    if (t == d.T[0] and y == d.Y[0])
                    else (v.H_HDS_TYR[d.T[-1], d.Y[d.Y.index(y)-1], r] if t == d.T[0] else v.H_HDS_TYR[t - 1, y, r]) * (1 - p.eta_HDS_self_R[r] * p.k__dT_)
                )
                + (
                    v.dH_HDS_in_TYR[t, y, r] * p.eta_HDS_ch_R[r]
                    - v.dH_HDS_out_TYR[t, y, r] / p.eta_HDS_dis_R[r]
                )
                * p.k__dT_
                for t in d.T
                for r in d.R
                for y in d.Y
            ),
            "HDS_balance",
        )
        c.dH_H2D_source_TYR["HDS"] = lambda t, y, r: v.dH_HDS_out_TYR[t, y, r]
        c.dH_H2D_sink_TYR["HDS"] = lambda t, y, r: v.dH_HDS_in_TYR[t, y, r]
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.H_HDS_CAPn_RY[r,y] >= v.H_HDS_CAPn_RY[r,d.Y[d.Y.index(y)-1]] for r in d.R for y in d.Y[1:]), "Installed power can only increase")
        c.C_TOT_inv_["HDS"] = quicksum((v.H_HDS_CAPn_RY[r,d.Y[-1]] * p.c_HDS_inv_R[r] * conv("€", "k€", 1e-3)) for r in d.R)
        c.C_TOT_invAnn_["HDS"] = quicksum((
            ((v.H_HDS_CAPn_RY[r,y] * p.c_HDS_inv_R[r] * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_HDS_R[r]))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for r in d.R for y in d.Y)
        c.C_TOT_RMI_["HDS"] = quicksum((quicksum((v.H_HDS_CAPn_RY[r,y] * p.c_HDS_inv_R[r] * conv("€", "k€", 1e-3)) * p.k_HDS_RMI_R[r] for r in d.R)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        if cf.main.reserve.consider == True:
            m.addConstrs((p.q_HDS_min_Y[y] <= (v.S_HDS_min_TY[t,y] + quicksum(v.H_HDS_TYR[t, y, r] for r in d.R)) for t in d.T[p.d_HDS_hours_cut_:] for y in d.Y), "Minimum load in storage")
            m.addConstrs(v.S_HDS_min_TY[t,y] == 0 for t in d.T[:p.d_HDS_hours_cut_] for y in d.Y)
            # Make it possible to also use the stored energy in case of emergency. Energy output of both derivative tanks is fast enough if storag size >= 24h reserve. However, conversion needs to be fast enough if NH3 is used:
            m.addConstrs((v.H_HDS_CAPn_RY["NH3",y]/p.d_HDS_min_reserve_hours_ <= v.dH_Con_CAPn_RRY["NH3","CH2",y] for y in d.Y), "Connect NH3 storage installation to sufficient ammoniac cracking")
        else:
            m.addConstrs(v.S_HDS_min_TY[t,y] == 0 for t in d.T for y in d.Y)

@dataclass
class Pip(Component):
    def param_func(self, sc: Scenario):
        sc.param("c_Pip_buy_Y", data=cf.comp.Pip.price_hydrogen, doc="Energy price", unit="€/kWh")
        sc.param("c_Pip_transport_Y", data=cf.comp.Pip.price_transport, doc="Price for transportation", unit="€/kWh")
        sc.param("c_Pip_inv_", data=cf.comp.Pip.CapEx, doc="Specific cost", unit="€/kW")
        sc.param("k_Pip_RMI_", data=cf.comp.Pip.RMI, doc=Descs.RMI.en)
        sc.param("N_Pip_", data=cf.comp.Pip.OL, doc=Etypes.N.en, unit="a")
        sc.param("eta_Pip_", data=cf.comp.Pip.eta, doc="Efficiency")
        sc.param("k_Pip_avail_", data=cf.comp.Pip.start_year, doc="Availability factor of pipeline installation")
        sc.param("k_Pip_feed_in_", data=cf.comp.Pip.feed_in, doc="Availability of feed into pipeline")
        if cf.comp.Pip.feed_in == True:
            sc.param("c_Pip_sell_Y", data=cf.comp.Pip.price_hydrogen_sell, doc="Energy sell price", unit="€/kWh")

        sc.var("dH_Pip_buy_TY", doc="Purchase power", unit="kW")
        sc.var("dH_Pip_out_TY", doc="Pipeline power considering Pipeline efficiency", unit="kW")
        if cf.comp.Pip.feed_in == True:
            sc.var("dH_Pip_sell_TY", doc="Selling power", unit="kW")
            sc.var("dH_Pip_in_TY", doc="Pipeline input power considering Pipeline efficiency", unit="kW")
        sc.var("dH_Pip_CAPn_Y", doc="New Capacity", unit="kW")
        sc.var("k_Pip_avail_Y", doc= "If pipeline is available")
    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(v.k_Pip_avail_Y[y] == 1 if (p.k_Pip_avail_ <= y) else v.k_Pip_avail_Y[y] == 0 for y in d.Y)
        m.addConstrs(
            ( 
                v.dH_Pip_buy_TY[t, y] <= v.k_Pip_avail_Y[y] * v.dH_Pip_CAPn_Y[y]
                for t in d.T
                for y in d.Y
            ),
            "Pip_limit",
        )
        m.addConstrs(
            (
                v.dH_Pip_out_TY[t, y] == v.dH_Pip_buy_TY[t, y] * p.eta_Pip_
                for t in d.T
                for y in d.Y
            ),
            "Pip_bal",
        )
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dH_Pip_CAPn_Y[y] >= v.dH_Pip_CAPn_Y[d.Y[d.Y.index(y)-1]] for y in d.Y[1:]), "Installed power can only increase")
        c.C_TOT_inv_["Pip"] = (v.dH_Pip_CAPn_Y[d.Y[-1]] * p.c_Pip_inv_ * conv("€", "k€", 1e-3))
        c.C_TOT_invAnn_["Pip"] = quicksum((((v.dH_Pip_CAPn_Y[y] * p.c_Pip_inv_ * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_Pip_))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["Pip"] = quicksum((((v.dH_Pip_CAPn_Y[y] * p.c_Pip_inv_ * conv("€", "k€", 1e-3)) * p.k_Pip_RMI_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        if p.k_Pip_feed_in_ == True:
            c.C_TOT_op_["Pip"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum(((v.dH_Pip_buy_TY.sum("*", y) * (p.c_Pip_buy_Y[y] + p.c_Pip_transport_Y[y])) - (v.dH_Pip_sell_TY.sum("*",y) * p.c_Pip_sell_Y[y]))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_)**(y - p.y_ref_)) for y in d.Y)
                * conv("€", "k€", 1e-3)
            )
        else:
            c.C_TOT_op_["Pip"] = (
                p.k__dT_
                * p.k__PartYearComp_
                * quicksum((v.dH_Pip_buy_TY.sum("*", y) * (p.c_Pip_buy_Y[y] + p.c_Pip_transport_Y[y]))/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_)**(y - p.y_ref_)) for y in d.Y)
                * conv("€", "k€", 1e-3)
            )
        c.dH_H2D_source_TYR["Pip"] = lambda t, y, r: v.dH_Pip_out_TY[t, y] if r == "CH2" else 0
        if p.k_Pip_feed_in_ == True:
            c.dH_H2D_sink_TYR["Pip"] = lambda t, y, r: v.dH_Pip_in_TY[t,y] if r == "CH2" else 0
            m.addConstrs(
            ( 
                v.dH_Pip_sell_TY[t, y] <= v.k_Pip_avail_Y[y] * v.dH_Pip_CAPn_Y[y]
                for t in d.T
                for y in d.Y
            ),
            "Pip_limit_feed_in",
            )
            m.addConstrs(
            (
                v.dH_Pip_in_TY[t, y] * p.eta_Pip_ == v.dH_Pip_sell_TY[t, y]
                for t in d.T
                for y in d.Y
            ),
            "Pip_bal_sell",
            )

@dataclass
class Lan(Component):
    """Landing of H2D"""

    def param_func(self, sc: Scenario):

        df = pd.DataFrame(cf.comp.Lan).T
        sc.param("c_Lan_buy_RY", data=df["price"], doc="Energy price", unit="€/kWh")
        sc.param("c_Lan_inv_R", data=df["CapEx"], doc="Specific cost", unit="€/kW")
        sc.param("k_Lan_RMI_R", data=df["RMI"], doc=Descs.RMI.en)
        sc.param("N_Lan_R", data=df["OL"], doc=Etypes.N.en, unit="a")
        sc.param("eta_Lan_R", data=df["eta"], doc="Efficiency")
        sc.param("p_Lan_profile_T", data=prep.get_lan_profile(), doc="Landing profile")
        sc.param("f_lan_frequency_days_Y", data=cf.main.landing_settings.frequency_days, doc="How often landing happens")
        sc.param("s_lan_ships_per_day_Y", data=cf.main.landing_settings.ships_per_day, doc="How many ships can be there at once")
        sc.param("p_lan_per_derivate_R", data=cf.main.landing_settings.maximum_power, doc="Maximum powers for discharging ships")

        sc.var("dH_Lan_buy_TYR", doc="Purchase power", unit="kW")
        sc.var("dH_Lan_out_TYR", doc="Landing power considering landing efficiency", unit="kW")
        sc.var("dH_Lan_CAPn_RY", doc="New Capacity", unit="kW")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        def p_Lan_profile_TY(t,y):
            ser = prep.get_lan_profile()
            if p.f_lan_frequency_days_Y[y] >= 2:
                for i in range(24,len(ser),(p.f_lan_frequency_days_Y[y]*24)):
                    ser[i:i+((p.f_lan_frequency_days_Y[y]*24)-25)] = 0
            return(ser[t])

        m.addConstrs(
            (
                v.dH_Lan_buy_TYR[t, y, r] == p_Lan_profile_TY(t,y) * v.dH_Lan_CAPn_RY[r,y]
                for t in d.T
                for r in d.R
                for y in d.Y
            ),
            "Lan_limit1",
        )
        m.addConstrs(
            (
                v.dH_Lan_CAPn_RY[r,y] <= p.p_lan_per_derivate_R[r] * p.s_lan_ships_per_day_Y[y]
                for r in d.R
                for y in d.Y
            ),
            "Lan_limit2",
        )
        m.addConstrs(
            (
                v.dH_Lan_out_TYR[t, y, r] == v.dH_Lan_buy_TYR[t, y, r] * p.eta_Lan_R[r]
                for t in d.T
                for r in d.R
                for y in d.Y
            ),
            "Lan_bal",
        )
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dH_Lan_CAPn_RY[r,y] >= v.dH_Lan_CAPn_RY[r,d.Y[d.Y.index(y)-1]] for r in d.R for y in d.Y[1:]), "Installed power can only increase")
        c.C_TOT_inv_["Lan"] = quicksum((v.dH_Lan_CAPn_RY[r,d.Y[-1]] * p.c_Lan_inv_R[r] * conv("€", "k€", 1e-3)) for r in d.R)
        c.C_TOT_invAnn_["Lan"] = quicksum((quicksum(
            (v.dH_Lan_CAPn_RY[r,y] * p.c_Lan_inv_R[r] * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_Lan_R[r]) for r in d.R
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_)**(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["Lan"] = quicksum((quicksum((v.dH_Lan_CAPn_RY[r,y] * p.c_Lan_inv_R[r] * conv("€", "k€", 1e-3)) * p.k_Lan_RMI_R[r] for r in d.R)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_op_["Lan"] = (
            p.k__dT_
            * p.k__PartYearComp_
            * quicksum((v.dH_Lan_buy_TYR.sum("*", y , r) * p.c_Lan_buy_RY[r][y])/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_)**(y - p.y_ref_)) for r in d.R for y in d.Y)
            * conv("€", "k€", 1e-3)
        )
        c.dH_H2D_source_TYR["Lan"] = lambda t, y, r: v.dH_Lan_out_TYR[t, y, r]

@dataclass
class Con(Component):
    """Hub-side H2D conversion"""

    def param_func(self, sc: Scenario):
        df = pd.concat({k: pd.DataFrame(v).T for k, v in cf.comp.Con.items()}, axis=0)
        sc.param("c_Con_inv_RR", data=df["CapEx"], doc="Specific cost", unit="€/kW")
        sc.param("k_Con_RMI_RR", data=df["RMI"], doc=Descs.RMI.en)
        sc.param("N_Con_RR", data=df["OL"], doc=Etypes.N.en, unit="a")
        sc.param("k_Con_el_RR", data=df["k_el"], doc="Specific elec. intensity based on output")
        sc.param("eta_Con_RR", data=df["eta"], doc="Efficiency")
        sc.var("dH_Con_CAPn_RRY", doc="New Capacity", unit="kW")
        sc.var("dH_Con_in_TYRR", doc="H2D conversion input power", unit="kW")
        sc.var("dH_Con_out_TYRR", doc="H2D conversion output power", unit="kW")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.dH_Con_out_TYRR[t, y, r, rr] <= v.dH_Con_CAPn_RRY[r, rr, y]
                for t in d.T
                for r in d.R
                for rr in d.R
                if r != rr
                for y in d.Y
            ),
            "Con_limit",
        )
        m.addConstrs(
            (
                v.dH_Con_out_TYRR[t, y, r, rr] == v.dH_Con_in_TYRR[t, y, r, rr] * p.eta_Con_RR[r, rr]
                for t in d.T
                for r in d.R
                for rr in d.R
                if r != rr
                for y in d.Y
            ),
            "Con_bal",
        )
        m.addConstrs(
            (v.dH_Con_out_TYRR[t, y, r, r] == 0 for t in d.T for r in d.R for y in d.Y), "Con_no_self_conversion"
        )
        c.dH_H2D_sink_TYR["Con"] = lambda t, y, r: v.dH_Con_in_TYRR.sum(t, y, r, "*")
        c.dH_H2D_source_TYR["Con"] = lambda t, y, r: v.dH_Con_out_TYRR.sum(t, y, "*", r)
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dH_Con_CAPn_RRY[r,rr,y] >= v.dH_Con_CAPn_RRY[r,rr,d.Y[d.Y.index(y)-1]] for r in d.R for rr in d.R for y in d.Y[1:] if r != rr), "Installed power can only increase")
        c.C_TOT_inv_["Con"] = quicksum((v.dH_Con_CAPn_RRY[r, rr, d.Y[-1]] * p.c_Con_inv_RR[r, rr] * conv("€", "k€", 1e-3)) for r in d.R for rr in d.R if r != rr)
        c.C_TOT_invAnn_["Con"] = quicksum((quicksum(
            (v.dH_Con_CAPn_RRY[r, rr, y] * p.c_Con_inv_RR[r, rr] * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_Con_RR[r, rr])
            for r in d.R
            for rr in d.R
            if r != rr
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["Con"] = quicksum((quicksum(
            (v.dH_Con_CAPn_RRY[r, rr, y] * p.c_Con_inv_RR[r, rr] * conv("€", "k€", 1e-3)) * p.k_Con_RMI_RR[r, rr] for r in d.R for rr in d.R if r != rr
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.P_EL_sink_TY["Con"] = lambda t,y: quicksum(
            v.dH_Con_out_TYRR[t, y, r, rr] * p.k_Con_el_RR[r, rr] for r in d.R for rr in d.R if r != rr
        )


@dataclass
class DCon(Component):
    """Demand-side H2D conversion"""

    def param_func(self, sc: Scenario):
        df = pd.concat({k: pd.DataFrame(v).T for k, v in cf.comp.Con.items()}, axis=0)
        sc.param("c_DCon_inv_RR", data=df["CapEx"], doc="Specific cost", unit="€/kW")
        sc.param("k_DCon_RMI_RR", data=df["RMI"], doc=Descs.RMI.en)
        sc.param("N_DCon_RR", data=df["OL"], doc=Etypes.N.en, unit="a")
        sc.param("k_DCon_el_RR", data=df["k_el"], doc="Specific elec. intensity")
        sc.param("eta_DCon_RR", data=df["eta"], doc="Efficiency")
        sc.var("dH_DCon_CAPn_ARRY", doc="New Capacity", unit="kW")
        sc.var("dH_DCon_in_TYARR", doc="H2D conversion input power", unit="kW")
        sc.var("dH_DCon_out_TYARR", doc="H2D conversion output power", unit="kW")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.dH_DCon_out_TYARR[t, y, a, r, rr] <= v.dH_DCon_CAPn_ARRY[a, r, rr, y]
                for t in d.T
                for a in d.A
                for r in d.R
                for rr in d.R
                if r != rr
                for y in d.Y
            ),
            "DCon_limit",
        )
        m.addConstrs(
            (
                v.dH_DCon_out_TYARR[t, y, a, r, rr]
                == v.dH_DCon_in_TYARR[t, y, a, r, rr] * p.eta_DCon_RR[r, rr]
                for t in d.T
                for a in d.A
                for r in d.R
                for rr in d.R
                if r != rr
                for y in d.Y
            ),
            "DCon_bal",
        )
        m.addConstrs(
            (v.dH_DCon_out_TYARR[t, y, a, r, r] == 0 for t in d.T for a in d.A for r in d.R for y in d.Y),
            "DCon_no_self_conversion",
        )
        c.dH_DH2D_sink_TYAR["DCon"] = lambda t, y, a, r: v.dH_DCon_in_TYARR.sum(t, y, a, r, "*")
        c.dH_DH2D_source_TYAR["DCon"] = lambda t, y, a, r: v.dH_DCon_out_TYARR.sum(t, y, a, "*", r)
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dH_DCon_CAPn_ARRY[a, r,rr,y] >= v.dH_DCon_CAPn_ARRY[a, r,rr,d.Y[d.Y.index(y)-1]] for a in d.A for r in d.R for rr in d.R for y in d.Y[1:] if r != rr), "Installed power can only increase")
        c.C_TOT_inv_["DCon"] = quicksum((v.dH_DCon_CAPn_ARRY.sum("*", r, rr, d.Y[-1]) * p.c_DCon_inv_RR[r, rr] * conv("€", "k€", 1e-3)) for r in d.R for rr in d.R if r != rr)
        c.C_TOT_invAnn_["DCon"] = quicksum((quicksum(
            (v.dH_DCon_CAPn_ARRY.sum("*", r, rr, y) * p.c_DCon_inv_RR[r, rr] * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_DCon_RR[r, rr])
            for r in d.R
            for rr in d.R
            if r != rr
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["DCon"] = quicksum((quicksum(
            (v.dH_DCon_CAPn_ARRY.sum("*", r, rr, y) * p.c_DCon_inv_RR[r, rr] * conv("€", "k€", 1e-3)) * p.k_DCon_RMI_RR[r, rr] for r in d.R for rr in d.R if r != rr
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_op_["DCon"] = (
            p.k__dT_
            * p.k__PartYearComp_
            * conv("€", "k€", 1e-3)
            * quicksum(
                v.dH_DCon_out_TYARR[t, y, a, r, rr]
                * p.k_DCon_el_RR[r, rr]
                * (p.c_EG_Y[y] + p.c_EG_addon_)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_))
                for t in d.T
                for a in d.A
                for r in d.R
                for rr in d.R
                if r != rr
                for y in d.Y
            )
        )


@dataclass
class Tra(Component):
    """Transport of H2D"""

    def param_func(self, sc: Scenario):
        df = pd.DataFrame(cf.comp.Tra.Truck).T
        sc.param("c_Tra_op_R", data=df["OpEx"], doc="Specific cost", unit="€/(kWh*1000km)")
        sc.param("c_Tra_inv_R", data=df["CapEx"], doc="Specific cost", unit="€/(kW*1000km)")
        sc.param("k_Tra_RMI_R", data=df["RMI"], doc=Descs.RMI.en)
        sc.param("N_Tra_R", data=df["OL"], doc=Etypes.N.en, unit="a")
        sc.param("eta_Tra_R", data=df["eta"], doc="Efficiency", unit="1/1000km")
        sc.param("s_Tra_A", data=util.get_s_Tra_A(), doc="Road distance to consumer a ∈ A")
        sc.var("dH_Tra_in_TYAR", doc="H2D transport power in", unit="kW")
        sc.var("dH_Tra_CAPn_ARY", doc="New capacity", unit="kW_H")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.dH_Tra_in_TYAR[t, y, a, r] <= v.dH_Tra_CAPn_ARY[a, r, y]
                for t in d.T
                for r in d.R
                for a in d.A
                for y in d.Y
            ),
            "Tra_LIM_dH_Tra_in_TAR",
        )
        if len(cf.projections.years) >= 2:
            m.addConstrs((v.dH_Tra_CAPn_ARY[a, r ,y] >= v.dH_Tra_CAPn_ARY[a, r, d.Y[d.Y.index(y)-1]] for a in d.A for r in d.R for y in d.Y[1:]), "Installed power can only increase")
        c.C_TOT_inv_["Tra"] = quicksum((quicksum(v.dH_Tra_CAPn_ARY[a, r, d.Y[-1]] * p.s_Tra_A[a]/1000 * p.c_Tra_inv_R[r] for a in d.A) * conv("€", "k€", 1e-3)) for r in d.R)
        c.C_TOT_invAnn_["Tra"] = quicksum((quicksum(
            (quicksum(v.dH_Tra_CAPn_ARY[a, r, y] * p.s_Tra_A[a]/1000 * p.c_Tra_inv_R[r] for a in d.A) * conv("€", "k€", 1e-3)) * get_annuity_factor(r=p.k__r_, N=p.N_Tra_R[r]) for r in d.R
        )/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_RMI_["Tra"] = quicksum((quicksum((quicksum(v.dH_Tra_CAPn_ARY[a, r, y] * p.s_Tra_A[a]/1000 * p.c_Tra_inv_R[r] for a in d.A) * conv("€", "k€", 1e-3)) * p.k_Tra_RMI_R[r] for r in d.R)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5)) / ((1+p.k__r_) **(y - p.y_ref_)) for y in d.Y)
        c.C_TOT_op_["Tra"] = quicksum(
            p.k__dT_
            * p.k__PartYearComp_
            * p.c_Tra_op_R[r]
            * p.s_Tra_A[a]/1000
            * v.dH_Tra_in_TYAR.sum("*", y, a, r)/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_))
            for a in d.A
            for r in d.R
            for y in d.Y
        ) * conv("€", "k€", 1e-3)
        c.dH_H2D_sink_TYR["Tra"] = lambda t, y, r: v.dH_Tra_in_TYAR.sum(t, y, "*", r)
        c.dH_DH2D_source_TYAR["Tra"] = lambda t, y, a, r: v.dH_Tra_in_TYAR[t, y, a, r]


@dataclass
class Dem(Component):
    """Hydrogen derivative (H2D) demand"""

    def dim_func(self, sc: Scenario):
        sc.dim("A", data=list(cf.consumer_data.keys()), doc="Considered consumers")
        sc.dim("R", data=cf.main.H2Ds, doc="Considered H2Ds")

    def param_func(self, sc: Scenario):
        sc.param(
            name="dH_Dem_TYAR",
            data=prep.dH_Dem_TYAR(sc),
            doc="H2D demand",
            unit="kW",
        )

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.dH_DH2D_sink_TYAR["Dem"] = lambda t, y, a, r: p.dH_Dem_TYAR[t, y, a, r] 

# @dataclass
# class FS(Component):
#     """Hydrogen filling stations (FS)"""

#     def param_func(self, sc: Scenario):
#         sc.param(
#             name="dH_FS_T",
#             data=prep.get_h2_station_demand(year=cf.projections.default_year),
#             doc="H2 demand for all filling stations",
#             unit="kW",
#         )

#     def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
#         c.dH_DH2D_sink_TYAR["DemFS"] = lambda t, y, a, r: p.dH_FS_T[t]


@dataclass
class DHN(Component):
    """District Heating Network"""

    def param_func(self, sc: Scenario):
        sc.param("c_DHN_sell_H", data=cf.comp.DHN.price, doc="Energy price", unit="€/kWh_th")
        sc.param("dQ_DHN_max_H", data=cf.comp.DHN.dQ_max, doc="Maximum", unit="kW_th")
        sc.param(
            "y_DHN_avail_TH",
            data=prep.y_DHN_avail_TH(sc, cf),
            doc="If the temperature level is available at a time",
        )
        sc.var("dQ_DHN_sell_TYH", doc="Sold thermal energy", unit="kW_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        m.addConstrs(
            (
                v.dQ_DHN_sell_TYH[t, y, h] <= p.y_DHN_avail_TH[t, h] * p.dQ_DHN_max_H[h]
                for t in d.T
                for h in d.H
                for y in d.Y
            )
        )
        c.C_TOT_op_["DHN"] = (
            p.k__dT_
            * p.k__PartYearComp_
            * -quicksum(v.dQ_DHN_sell_TYH[t, y, h] * p.c_DHN_sell_H[h]/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_)) for t in d.T for h in d.H for y in d.Y)
            * conv("€", "k€", 1e-3)
        )
        c.dQ_heating_sink_TYH["DHN"] = (
            lambda t, y, h: v.dQ_DHN_sell_TYH[t, y, h] if h.startswith("DHN") else 0
        )

@dataclass
class REC(Component):
    """Recooler for ambient and rhine"""

    def param_func(self, sc: Scenario):
        sc.param("c_REC_amb_", data=cf.comp.REC.amb.cost, doc="Cooling/heating from ambient temperatures", unit="€/kWh_th")
        sc.param("c_REC_rhine_", data=cf.comp.REC.rhine.cost, doc="Cooling/heating from rhine temperatures", unit="€/kWh_th")

    def model_func(self, sc: Scenario, m: Model, d: Dimensions, p: Params, v: Vars, c: Collectors):
        c.C_TOT_op_["REC_amb_sink"] = quicksum(quicksum(x(t, y) for x in c.dQ_amb_sink_TY.values())/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_)) for t in d.T for y in d.Y) * p.c_REC_amb_ * p.k__PartYearComp_ * p.k__dT_ * conv("€", "k€", 1e-3)
        c.C_TOT_op_["REC_amb_source"] = quicksum(quicksum(x(t, y) for x in c.dQ_amb_source_TY.values())/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_)) for t in d.T for y in d.Y) * p.c_REC_amb_ * p.k__PartYearComp_ * p.k__dT_ * conv("€", "k€", 1e-3)
        c.C_TOT_op_["REC_rhine_sink"] = quicksum(quicksum(x(t, y) for x in c.dQ_rhine_sink_TY.values())/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_)) for t in d.T for y in d.Y) * p.c_REC_rhine_ * p.k__PartYearComp_ * p.k__dT_ * conv("€", "k€", 1e-3)
        c.C_TOT_op_["REC_rhine_source"] = quicksum(quicksum(x(t, y) for x in c.dQ_rhine_source_TY.values())/get_annuity_factor(r=p.k__r_, N= 3 if y == 2027 else 5) / ((1+p.k__r_) **(y - p.y_ref_)) for t in d.T for y in d.Y) * p.c_REC_rhine_ * p.k__PartYearComp_ * p.k__dT_ * conv("€", "k€", 1e-3)

order_restrictions = [
    ("EG", {"PV", "WT", "PPA"}),  # EG collects P_EG_sell_T
    ("BES", {}),
    ("OnPV", {}),
    ("PV", {}),
    ("WT", {}),
    ("PPA", {}),
    ("EHP", {}),
    ("TES", {"EHP"}),
    ("Elc", {}),
    ("HDS", {}),
    ("Pip", {}),
    ("Lan", {}),
    ("Con", {}),
    ("DCon", {}),
    ("Tra", {}),
    ("Dem", {}),
    ("DHN", {}),
    ("REC", {}),
]
order_restrictions.append(("Main", [x[0] for x in order_restrictions]))

set_component_order_by_order_restrictions(order_restrictions=order_restrictions, classes=globals())
