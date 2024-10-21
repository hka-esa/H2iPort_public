import pandas as pd
from draf import Scenario
from draf import helper as hp

from h2iport.config import Config
from h2iport.paths import DATA_DIR

cf = Config.cf


def write_equally_distributed_profile(steps=8760):
    ser = pd.Series(1, range(steps)).rename("profile")
    fp = DATA_DIR / "dummy/profiles/equally_distributed.csv"
    hp.write(ser, fp)


def write_slightly_seasonal_profile():
    sc_helper = Scenario(freq="60min", year=2019, country="DE")
    ser = sc_helper.dated(sc_helper.prep.dQ_hDem_T()).resample("M").mean().resample("H").ffill()
    ser = ser.reindex(sc_helper.dtindex).ffill().bfill()
    ser = 3 + ser / ser.max()
    ser = ser.reset_index(drop=True)
    ser = ser.round(4).rename("profile")
    fp = DATA_DIR / "dummy/profiles/slightly_seasonal.csv"
    hp.write(ser, fp)


def get_s_Tra_A():
    """Get road distance to consumer a ∈ A."""
    return pd.DataFrame(cf.consumer_data).T["road_distance"]


def my_sankey_builder(self) -> str:
    templates = {
        "P_EL_source_TY": "E {k} el_hub {v}",
        "P_EL_sink_TY": "E el_hub {k} {v}",
        "dQ_cooling_source_TYN": "C {k} cool_hub {v}",
        "dQ_cooling_sink_TYN": "C cool_hub {k} {v}",
        "dQ_heating_source_TYH": "Q {k} heat_hub {v}",
        "dQ_heating_sink_TYH": "Q heat_hub {k} {v}",
        "F_fuel_F": "F FUEL {k} {v}",
        "dQ_amb_sink_TY": "M ambient_source {k} {v}",
        "dQ_rhine_sink_TY": "M rhine_source {k} {v}",
        "dQ_amb_source_TY": "C {k} ambient_sink {v}",
        "dQ_rhine_source_TY": "C {k} rhine_sink {v}",
        "dH_H2D_source_TYR": "H {k} H2D {v}",
        "dH_H2D_sink_TYR": "H H2D {k} {v}",
        "dH_DH2D_source_TYAR": "H {k} DH2D {v}",
        "dH_DH2D_sink_TYAR": "H DH2D {k} {v}",
    }
    header = ["type source target value"]
    rows = [
        templates[name].format(k=k, v=v)
        for name, collector in self.get_all_collector_values().items()
        for k, v in collector.items()
        if name in templates
    ]
    return "\n".join(header + rows)

def my_collector_values(self):
    try: # with pipeline feed in
        return(
        {'P_EG_sell_TY':{"OnPV":sum(self.res.P_OnPV_FI_TY[:,self.year_sankey]),
        "PV":sum(self.res.P_PV_FI_TY[:,self.year_sankey]),
        "WT":sum(self.res.P_WT_FI_TY[:,self.year_sankey]),
        "PV_PPA":sum(self.res.P_PV_PPA_FI_TY[:,self.year_sankey]),
        "WTon_PPA":sum(self.res.P_WTon_PPA_FI_TY[:,self.year_sankey]),
        "WToff_PPA":sum(self.res.P_WToff_PPA_FI_TY[:,self.year_sankey])},
        'P_EL_source_TY':{"BES":sum(self.res.P_BES_out_TY[:,self.year_sankey]),
        "OnPV":sum(self.res.P_OnPV_FI_TY[:,self.year_sankey]+self.res.P_OnPV_OC_TY[:,self.year_sankey]),
        "PV":sum(self.res.P_PV_FI_TY[:,self.year_sankey]+self.res.P_PV_OC_TY[:,self.year_sankey]),
        "WT":sum(self.res.P_WT_FI_TY[:,self.year_sankey]+self.res.P_WT_OC_TY[:,self.year_sankey]),
        "PV_PPA":sum(self.res.P_PV_PPA_FI_TY[:,self.year_sankey]+self.res.P_PV_PPA_OC_TY[:,self.year_sankey]),
        "WTon_PPA":sum(self.res.P_WTon_PPA_FI_TY[:,self.year_sankey]+self.res.P_WTon_PPA_OC_TY[:,self.year_sankey]),
        "WToff_PPA":sum(self.res.P_WToff_PPA_FI_TY[:,self.year_sankey]+self.res.P_WToff_PPA_OC_TY[:,self.year_sankey]),
        "EG":sum(self.res.P_EG_buy_TY[:,self.year_sankey])},
        'P_EL_sink_TY':{"BES":sum(self.res.P_BES_in_TY[:,self.year_sankey]),
        "EHP":sum(self.res.P_EHP_TYEC[:,self.year_sankey,:,:]),
        "Elc":sum(self.res.P_Elc_TY[:,self.year_sankey]),
        "Con":(sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,"CH2","NH3"])*self.params.k_Con_el_RR["CH2","NH3"]+sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,"NH3","CH2"])*self.params.k_Con_el_RR["NH3","CH2"]),
        "EG":sum(self.res.P_EG_sell_TY[:,self.year_sankey])},
        'dQ_heating_source_TYH':{"EHP":sum(self.res.dQ_EHP_Cond_TYEC[:,self.year_sankey,:,:])},
        'dQ_heating_sink_TYH':{"DHN":sum(self.res.dQ_DHN_sell_TYH[:,self.year_sankey,:])},
        'dQ_cooling_source_TYN':{"TES":sum(self.res.dQ_TES_out_TYL[:,self.year_sankey,:]),
        "Elc":sum(self.res.dQ_Elc_HP_TY[:,self.year_sankey])},
        'dQ_cooling_sink_TYN':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"Elc",:]),
        "TES":sum(self.res.dQ_TES_in_TYL[:,self.year_sankey,:])},
        'dQ_amb_sink_TY':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"amb",:])},
        'dQ_rhine_sink_TY':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"rhine_water",:])},
        'dQ_amb_source_TY':{"Elc":sum(self.res.dQ_Elc_amb_TY[:,self.year_sankey])},
        'dQ_rhine_source_TY':{"Elc":sum(self.res.dQ_Elc_rhine_TY[:,self.year_sankey])},
        'dH_H2D_source_TYR':{"Elc":sum(self.res.dH_Elc_TY[:,self.year_sankey]),
        "HDS":sum(self.res.dH_HDS_out_TYR[:,self.year_sankey,:]),
        "Pip":sum(self.res.dH_Pip_out_TY[:,self.year_sankey]),
        "Lan":sum(self.res.dH_Lan_out_TYR[:,self.year_sankey,:]),
        "Con":sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,:,:])},
        'dH_H2D_sink_TYR':{"HDS":sum(self.res.dH_HDS_in_TYR[:,self.year_sankey,:]),
        "Con":sum(self.res.dH_Con_in_TYRR[:,self.year_sankey,:,:]),
        "Tra":sum(self.res.dH_Tra_in_TYAR[:,self.year_sankey,:,:]),
        "Pip":sum(self.res.dH_Pip_in_TY[:,self.year_sankey])},
        'dH_DH2D_source_TYAR':{"SR_MiRO":(self.res.dH_SR_Y[self.year_sankey]*8760),
        "Tra":sum(self.res.dH_Tra_in_TYAR[:,self.year_sankey,:,:])},
        'dH_DH2D_sink_TYAR':{"HDSMiRO":sum(self.res.dH_HDSMiRO_in_TYR[:,self.year_sankey,:]),
        "Dem":sum(self.params.dH_Dem_TYAR[:,self.year_sankey,:,:])}})
    except:
        return(
        {'P_EG_sell_TY':{"OnPV":sum(self.res.P_OnPV_FI_TY[:,self.year_sankey]),
        "PV":sum(self.res.P_PV_FI_TY[:,self.year_sankey]),
        "WT":sum(self.res.P_WT_FI_TY[:,self.year_sankey]),
        "PV_PPA":sum(self.res.P_PV_PPA_FI_TY[:,self.year_sankey]),
        "WTon_PPA":sum(self.res.P_WTon_PPA_FI_TY[:,self.year_sankey]),
        "WToff_PPA":sum(self.res.P_WToff_PPA_FI_TY[:,self.year_sankey])},
        'P_EL_source_TY':{"BES":sum(self.res.P_BES_out_TY[:,self.year_sankey]),
        "OnPV":sum(self.res.P_OnPV_FI_TY[:,self.year_sankey]+self.res.P_OnPV_OC_TY[:,self.year_sankey]),
        "PV":sum(self.res.P_PV_FI_TY[:,self.year_sankey]+self.res.P_PV_OC_TY[:,self.year_sankey]),
        "WT":sum(self.res.P_WT_FI_TY[:,self.year_sankey]+self.res.P_WT_OC_TY[:,self.year_sankey]),
        "PV_PPA":sum(self.res.P_PV_PPA_FI_TY[:,self.year_sankey]+self.res.P_PV_PPA_OC_TY[:,self.year_sankey]),
        "WTon_PPA":sum(self.res.P_WTon_PPA_FI_TY[:,self.year_sankey]+self.res.P_WTon_PPA_OC_TY[:,self.year_sankey]),
        "WToff_PPA":sum(self.res.P_WToff_PPA_FI_TY[:,self.year_sankey]+self.res.P_WToff_PPA_OC_TY[:,self.year_sankey]),
        "EG":sum(self.res.P_EG_buy_TY[:,self.year_sankey])},
        'P_EL_sink_TY':{"BES":sum(self.res.P_BES_in_TY[:,self.year_sankey]),
        "EHP":sum(self.res.P_EHP_TYEC[:,self.year_sankey,:,:]),
        "Elc":sum(self.res.P_Elc_TY[:,self.year_sankey]),
        "Con":(sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,"CH2","NH3"])*self.params.k_Con_el_RR["CH2","NH3"]+sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,"NH3","CH2"])*self.params.k_Con_el_RR["NH3","CH2"]),
        "EG":sum(self.res.P_EG_sell_TY[:,self.year_sankey])},
        'dQ_heating_source_TYH':{"EHP":sum(self.res.dQ_EHP_Cond_TYEC[:,self.year_sankey,:,:])},
        'dQ_heating_sink_TYH':{"DHN":sum(self.res.dQ_DHN_sell_TYH[:,self.year_sankey,:])},
        'dQ_cooling_source_TYN':{"TES":sum(self.res.dQ_TES_out_TYL[:,self.year_sankey,:]),
        "Elc":sum(self.res.dQ_Elc_HP_TY[:,self.year_sankey])},
        'dQ_cooling_sink_TYN':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"Elc",:]),
        "TES":sum(self.res.dQ_TES_in_TYL[:,self.year_sankey,:])},
        'dQ_amb_sink_TY':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"amb",:])},
        'dQ_rhine_sink_TY':{"EHP":sum(self.res.dQ_EHP_Eva_TYEC[:,self.year_sankey,"rhine_water",:])},
        'dQ_amb_source_TY':{"Elc":sum(self.res.dQ_Elc_amb_TY[:,self.year_sankey])},
        'dQ_rhine_source_TY':{"Elc":sum(self.res.dQ_Elc_rhine_TY[:,self.year_sankey])},
        'dH_H2D_source_TYR':{"Elc":sum(self.res.dH_Elc_TY[:,self.year_sankey]),
        "HDS":sum(self.res.dH_HDS_out_TYR[:,self.year_sankey,:]),
        "Pip":sum(self.res.dH_Pip_out_TY[:,self.year_sankey]),
        "Lan":sum(self.res.dH_Lan_out_TYR[:,self.year_sankey,:]),
        "Con":sum(self.res.dH_Con_out_TYRR[:,self.year_sankey,:,:])},
        'dH_H2D_sink_TYR':{"HDS":sum(self.res.dH_HDS_in_TYR[:,self.year_sankey,:]),
        "Con":sum(self.res.dH_Con_in_TYRR[:,self.year_sankey,:,:]),
        "Tra":sum(self.res.dH_Tra_in_TYAR[:,self.year_sankey,:,:])},
        'dH_DH2D_source_TYAR':{"SR_MiRO":(self.res.dH_SR_Y[self.year_sankey]*8760),
        "Tra":sum(self.res.dH_Tra_in_TYAR[:,self.year_sankey,:,:])},
        'dH_DH2D_sink_TYAR':{"HDSMiRO":sum(self.res.dH_HDSMiRO_in_TYR[:,self.year_sankey,:]),
        "Dem":sum(self.params.dH_Dem_TYAR[:,self.year_sankey,:,:])}})

def get_price_per_year(self, year = 2027):
        # Store results:
        C_Vals = {}
        C_TOT_op_ = {}
        C_TOT_invAnn_ = {}
        C_TOT_RMI_ = {}
        DH_Dem = {}

        # Operation:
        C_TOT_op_["EG_peak"] = (self.res.P_EG_buyPeak_Y[year] * self.params.c_EG_buyPeak_ * 1/1000)
        try:
            C_TOT_op_["EG_var"] = sum(self.res.P_EG_buy_TY[t,year] * (self.params.c_EG_T[t] + self.params.c_EG_addon_) - self.res.P_EG_sell_TY[t,year] * self.params.c_EG_sell_ for t in self.dims.T)*1/1000 # old version, before use of grid allowed
        except:
            C_TOT_op_["EG_fix"] = sum(self.res.P_EG_buy_TY[t,year] * (self.params.c_EG_Y[year] + self.params.c_EG_addon_) - self.res.P_EG_sell_TY[t,year] * self.params.c_EG_sell_ for t in self.dims.T)*1/1000 # new version
        C_TOT_op_["PV_PPA"] = self.params.c_PV_el_ * 1/1000 * sum(self.res.P_PV_buy_cap_Y[year] * self.params.p_PV_profile_T[t] for t in self.dims.T)
        C_TOT_op_["WTon_PPA"] = self.params.c_WTon_el_ * 1/1000 * sum(self.res.P_WTon_buy_cap_Y[year] * self.params.p_WTon_profile_T[t] for t in self.dims.T)
        C_TOT_op_["WToff_PPA"] = self.params.c_WToff_el_ * 1/1000 * sum(self.res.P_WToff_buy_cap_Y[year] * self.params.p_WToff_profile_T[t] for t in self.dims.T)
        C_TOT_op_["PV"] = sum(self.res.P_PV_OC_TY[t,year] for t in self.dims.T) * self.params.c_EG_addon_ * 1/1000
        C_TOT_op_["WT"] = sum(self.res.P_WT_OC_TY[t,year] for t in self.dims.T) * self.params.c_EG_addon_ * 1/1000
        C_TOT_op_["Elc"] = self.params.c_Elc_water_ * 1/1000 * 1/1000 * sum(self.res.M_Elc_water_TY[t,year] for t in self.dims.T)
        try:
            C_TOT_op_["Pip"] = ((sum(self.res.dH_Pip_buy_TY[t,year] for t in self.dims.T) * (self.params.c_Pip_buy_Y[year] + self.params.c_Pip_transport_Y[year])) - (sum(self.res.dH_Pip_sell_TY[t,year] for t in self.dims.T) * self.params.c_Pip_sell_Y[year]))* 1/1000 # new version with input
        except:
            C_TOT_op_["Pip"] = sum(self.res.dH_Pip_buy_TY[t,year] for t in self.dims.T) * (self.params.c_Pip_buy_Y[year] + self.params.c_Pip_transport_Y[year])* 1/1000 # old version or without input
        C_TOT_op_["Lan"] = sum((sum(self.res.dH_Lan_buy_TYR[t,year,r] for t in self.dims.T) * self.params.c_Lan_buy_RY[r][year]) for r in self.dims.R) * 1/1000
        #C_TOT_op_["DCon"] = sum(self.res.dH_DCon_out_TYARR[t, year, a, r, rr] * self.params.k_DCon_el_RR[r, rr] * (self.params.c_EG_T[t] + self.params.c_EG_addon_) for t in self.dims.T for a in self.dims.A for r in self.dims.R for rr in self.dims.R if r != rr) * 1/1000
        C_TOT_op_["Tra"] = sum(self.params.c_Tra_op_R[r] * self.params.s_Tra_A[a]/1000 * sum(self.res.dH_Tra_in_TYAR[t,year,a,r] for t in self.dims.T)  for a in self.dims.A for r in self.dims.R) *1/1000
        C_TOT_op_["DHN"] = -sum(self.res.dQ_DHN_sell_TYH[t, year, h] * self.params.c_DHN_sell_H[h]  for t in self.dims.T for h in self.dims.H) * 1/1000
        C_TOT_op_["REC_amb_sink"] = sum(self.res.dQ_EHP_Eva_TYEC[t,year,"amb",c]  for t in self.dims.T for c in self.dims.C) * self.params.c_REC_amb_ *1/1000                  
        C_TOT_op_["REC_amb_source"] = sum(self.res.dQ_Elc_amb_TY[t,year] for t in self.dims.T) * self.params.c_REC_amb_ *1/1000         
        C_TOT_op_["REC_rhine_sink"] = sum(self.res.dQ_EHP_Eva_TYEC[t,year,"rhine_water",c] for t in self.dims.T for c in self.dims.C) * self.params.c_REC_rhine_ *1/1000
        C_TOT_op_["REC_rhine_source"] = sum(self.res.dQ_Elc_rhine_TY[t,year] for t in self.dims.T) * self.params.c_REC_rhine_ *1/1000
        # Invest:
        C_TOT_invAnn_["BES"] = self.res.E_BES_CAPn_Y[year] * self.params.c_BES_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_BES_)
        C_TOT_invAnn_["OnPV"] = self.res.P_OnPV_CAPn_Y[year] * self.params.c_OnPV_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_OnPV_)
        C_TOT_invAnn_["PV"] = self.res.P_PV_CAPn_Y[year] * self.params.c_PV_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_PV_)
        C_TOT_invAnn_["WT"] = self.res.P_WT_CAPn_Y[year] * self.params.c_WT_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_WT_)
        C_TOT_invAnn_["EHP"] = self.res.dQ_EHP_CAPn_Y[year] * self.params.c_EHP_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_EHP_)
        C_TOT_invAnn_["TES"] = sum((self.res.Q_TES_CAPn_LY[l,year] * self.params.c_TES_inv_ *1/1000) for l in self.dims.L) * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_TES_)
        C_TOT_invAnn_["Elc"] = self.res.P_Elc_CAPn_Y[year] * self.params.c_Elc_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_Elc_)
        C_TOT_invAnn_["HDS"] = sum((self.res.H_HDS_CAPn_RY[r,year] * self.params.c_HDS_inv_R[r] *1/1000)  * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_HDS_R[r]) for r in self.dims.R)
        C_TOT_invAnn_["Pip"] = self.res.dH_Pip_CAPn_Y[year] * self.params.c_Pip_inv_ *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_Pip_)
        C_TOT_invAnn_["Lan"] = sum((self.res.dH_Lan_CAPn_RY[r,year] * self.params.c_Lan_inv_R[r] *1/1000) * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_Lan_R[r]) for r in self.dims.R)
        C_TOT_invAnn_["Con"] = sum((self.res.dH_Con_CAPn_RRY[r, rr, year] * self.params.c_Con_inv_RR[r, rr] *1/1000) * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_Con_RR[r, rr]) for r in self.dims.R for rr in self.dims.R if r != rr)
        #C_TOT_invAnn_["DCon"] = sum((sum(self.res.dH_DCon_CAPn_ARRY[a,r,rr,year] for a in self.dims.A) * self.params.c_DCon_inv_RR[r, rr] *1/1000) * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_DCon_RR[r, rr]) for r in self.dims.R for rr in self.dims.R if r != rr)
        C_TOT_invAnn_["Tra"] = sum(sum(self.res.dH_Tra_CAPn_ARY[a, r, year] * self.params.s_Tra_A[a]/1/1000 * self.params.c_Tra_inv_R[r] for a in self.dims.A) *1/1000 * hp.get_annuity_factor(r=self.params.k__r_, N=self.params.N_Tra_R[r]) for r in self.dims.R)
        # RMI:
        C_TOT_RMI_["BES"] = (self.res.E_BES_CAPn_Y[year] * self.params.c_BES_inv_ *1/1000) * self.params.k_BES_RMI_
        C_TOT_RMI_["OnPV"] = (self.res.P_OnPV_CAPn_Y[year] * self.params.c_OnPV_inv_ *1/1000) * self.params.k_OnPV_RMI_
        C_TOT_RMI_["PV"] = (self.res.P_PV_CAPn_Y[year] * self.params.c_PV_inv_ *1/1000) * self.params.k_PV_RMI_
        C_TOT_RMI_["WT"] = (self.res.P_WT_CAPn_Y[year] * self.params.c_WT_inv_ *1/1000) * self.params.k_WT_RMI_
        C_TOT_RMI_["EHP"] = (self.res.dQ_EHP_CAPn_Y[year] * self.params.c_EHP_inv_ *1/1000) * self.params.k_EHP_RMI_
        C_TOT_RMI_["TES"] = sum((self.res.Q_TES_CAPn_LY[l,year] * self.params.c_TES_inv_ *1/1000) * self.params.k_TES_RMI_ for l in self.dims.L)
        C_TOT_RMI_["Elc"] = (self.res.P_Elc_CAPn_Y[year] * self.params.c_Elc_inv_ *1/1000) * self.params.k_Elc_RMI_
        C_TOT_RMI_["HDS"] = sum((self.res.H_HDS_CAPn_RY[r,year] * self.params.c_HDS_inv_R[r] *1/1000) * self.params.k_HDS_RMI_R[r] for r in self.dims.R)
        C_TOT_RMI_["Pip"] = (self.res.dH_Pip_CAPn_Y[year] * self.params.c_Pip_inv_ *1/1000) * self.params.k_Pip_RMI_
        C_TOT_RMI_["Lan"] = sum((self.res.dH_Lan_CAPn_RY[r,year] * self.params.c_Lan_inv_R[r] *1/1000) * self.params.k_Lan_RMI_R[r] for r in self.dims.R)
        C_TOT_RMI_["Con"] = sum((self.res.dH_Con_CAPn_RRY[r, rr, year] * self.params.c_Con_inv_RR[r, rr] *1/1000) * self.params.k_Con_RMI_RR[r, rr] for r in self.dims.R for rr in self.dims.R if r != rr)
        #C_TOT_RMI_["DCon"] = sum((sum(self.res.dH_DCon_CAPn_ARRY[a,r,rr,year] for a in self.dims.A) * self.params.c_DCon_inv_RR[r, rr] *1/1000) * self.params.k_DCon_RMI_RR[r, rr] for r in self.dims.R for rr in self.dims.R if r != rr)
        C_TOT_RMI_["Tra"] = sum((sum(self.res.dH_Tra_CAPn_ARY[a, r, year] * self.params.s_Tra_A[a]/1/1000 * self.params.c_Tra_inv_R[r] for a in self.dims.A) *1/1000) * self.params.k_Tra_RMI_R[r] for r in self.dims.R)

        # Production for comparison:
        DH_Dem["Total"] = sum(self.params.dH_Dem_TYAR[:,year,:,:])

        # Calculation:
        C_TOT_op_sum_ = sum(C_TOT_op_.values())
        C_TOT_invAnn_sum_ = sum(C_TOT_invAnn_.values())
        C_TOT_RMI_sum_ = sum(C_TOT_RMI_.values())

        C_Vals["C_TOT_op_"] = C_TOT_op_
        C_Vals["C_TOT_op_sum_"] = C_TOT_op_sum_
        C_Vals["C_TOT_invAnn_"] = C_TOT_invAnn_
        C_Vals["C_TOT_invAnn_sum_"] =C_TOT_invAnn_sum_
        C_Vals["C_TOT_RMI_"] = C_TOT_RMI_
        C_Vals["C_TOT_RMI_sum_"] = C_TOT_RMI_sum_
        C_Vals["DH_Dem"] = DH_Dem
        C_Vals["year"] = year

        return C_Vals

