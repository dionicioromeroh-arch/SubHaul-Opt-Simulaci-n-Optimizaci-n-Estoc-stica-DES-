"""
SubHaul-Opt — Módulo Prescriptivo (Sim-Opt), Estadística Formal, Benchmark CRN y Criterios ABET.
Feria ABET UNI 2026-II | Métodos de Explotación Subterránea (MI753-A)

Dependencias: motor_des.py, numpy, pandas, scipy.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from motor_des import (
    M3_MIN_TO_CFM,
    POLICY_LABELS,
    POLICY_OPTIONS,
    BreakdownConfig,
    SimulationConfig,
    SubHaulDESEngine,
    build_huancapeti_base_config,
    normalize_policy,
)


# ==============================================================================
# 1. UTILIDADES ESTADÍSTICAS (IC t-STUDENT Y FORMATO)
# ==============================================================================
def compute_mean_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calcula media, desviación estándar, error estándar e IC t-Student."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "se": 0.0, "ci_half": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    mean = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "se": 0.0, "ci_half": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1))
    ci_half = t_crit * se
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_half": ci_half,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
    }


def format_assignment(assignment: Dict[str, int], zone_order: Sequence[str]) -> str:
    """Formatea un diccionario de flota como '6-2-7' según el orden de zonas."""
    return "-".join(str(int(assignment.get(zid, 0))) for zid in zone_order)


def _format_hours_hm(hours: float) -> str:
    """Formatea horas decimales como '6 h 18 min (6.30 h)'."""
    total_min = int(round(max(0.0, float(hours)) * 60.0))
    hh, mm = divmod(total_min, 60)
    return f"{hh} h {mm:02d} min ({hours:.2f} h)"


# ==============================================================================
# 2. MÓDULO ABET: VENTILACIÓN (D.S. 023-2017-EM) + DIÉSEL / CO2 EN COLAS
# ==============================================================================
def compute_abet_metrics(
    config: SimulationConfig,
    kpi_summary: Dict[str, object],
    baseline_wait_hr: Optional[float] = None,
) -> Dict[str, object]:
    """
    Calcula requerimientos normativos de ventilación (D.S. 023-2017-EM) bajo pico
    simultáneo local de volquetes en interior mina por zona (Fu = 1.0, sin promediar
    en la guardia y sin aplicar Dm como factor reductor), junto con el trío
    operativo-ambiental de combustible desperdiciado y CO2 en colas de volquetes
    (excluyendo completamente el tiempo idle/standby de scooptrams).

      Q_pico,zona = factor_vent * (N_scoop,zona * HP_scoop,zona + N_volq_int_max,zona * HP_volq)
      Q_total     = sum_zona Q_pico,zona
    """
    g = config.global_cfg
    active_zones = [z for z in config.zones if z.active]

    n_trucks = int(kpi_summary["n_trucks_total"])
    truck_fu = float(kpi_summary.get("truck_fu", 0.0))
    dm_volq = float(g.dm_volquete)
    f_vent = float(g.factor_vent_m3_min_hp)
    hp_volq = float(g.hp_volquete)

    peak_by_zone_in: Dict[str, int] = kpi_summary.get("underground_trucks_peak_by_zone", {})
    avg_by_zone_in: Dict[str, float] = kpi_summary.get("underground_trucks_avg_by_zone", {})

    underground_trucks_peak_by_zone: Dict[str, int] = {}
    underground_trucks_avg_by_zone: Dict[str, float] = {}
    vent_q_peak_m3_min_by_zone: Dict[str, float] = {}
    vent_q_peak_cfm_by_zone: Dict[str, float] = {}
    vent_cap_max_m3_min_by_zone: Dict[str, float] = {}
    vent_cap_usage_pct_by_zone: Dict[str, float] = {}
    vent_margin_m3_min_by_zone: Dict[str, Optional[float]] = {}
    vent_alert_by_zone: Dict[str, bool] = {}

    total_scoops = 0
    for z in active_zones:
        zid = z.zone_id
        hp_sc = float(z.hp_scoop) if z.hp_scoop is not None else float(g.hp_scoop)
        peak_volq = int(peak_by_zone_in.get(zid, z.n_trucks))
        avg_volq = float(avg_by_zone_in.get(zid, 0.0))

        q_peak_m3 = f_vent * (int(z.n_scoops) * hp_sc + peak_volq * hp_volq)
        q_peak_cfm = q_peak_m3 * M3_MIN_TO_CFM

        underground_trucks_peak_by_zone[zid] = peak_volq
        underground_trucks_avg_by_zone[zid] = round(avg_volq, 4)
        vent_q_peak_m3_min_by_zone[zid] = round(q_peak_m3, 2)
        vent_q_peak_cfm_by_zone[zid] = round(q_peak_cfm, 1)

        cap_z = float(z.cap_max_ventilacion_m3_min)
        vent_cap_max_m3_min_by_zone[zid] = round(cap_z, 2)
        if cap_z > 0.0:
            vent_cap_usage_pct_by_zone[zid] = round((q_peak_m3 / cap_z) * 100.0, 2)
            vent_margin_m3_min_by_zone[zid] = round(cap_z - q_peak_m3, 2)
            vent_alert_by_zone[zid] = bool(q_peak_m3 > cap_z)
        else:
            vent_cap_usage_pct_by_zone[zid] = 0.0
            vent_margin_m3_min_by_zone[zid] = None
            vent_alert_by_zone[zid] = False

        total_scoops += int(z.n_scoops)

    vent_q_peak_total_m3_min = round(sum(vent_q_peak_m3_min_by_zone.values()), 2)
    vent_q_peak_total_cfm = round(vent_q_peak_total_m3_min * M3_MIN_TO_CFM, 1)
    vent_any_alert = any(vent_alert_by_zone.values())

    # Trío operativo-ambiental: EXCLUSIVAMENTE horas en cola de volquetes (sin idle de scooptrams)
    guardias = int(g.guardias_anio)
    wait_hr = float(kpi_summary["total_truck_wait_hr"])
    diesel_wasted_wait_l_shift = wait_hr * float(g.diesel_ralenti_l_hr)
    diesel_wasted_wait_l_year = diesel_wasted_wait_l_shift * guardias
    diesel_wasted_wait_cost_usd_shift = diesel_wasted_wait_l_shift * float(g.precio_diesel_usd_l)
    diesel_wasted_wait_cost_usd_year = diesel_wasted_wait_cost_usd_shift * guardias
    co2_wasted_wait_kg_shift = diesel_wasted_wait_l_shift * float(g.factor_co2_kg_l)
    co2_wasted_wait_t_year = (co2_wasted_wait_kg_shift * guardias) / 1000.0

    # Ahorros / emisiones evitadas respecto a una política base (ej. FIJA)
    wait_saved_hr = (float(baseline_wait_hr) - wait_hr) if baseline_wait_hr is not None else 0.0
    diesel_saved_l_shift = wait_saved_hr * float(g.diesel_ralenti_l_hr)
    diesel_saved_l_year = diesel_saved_l_shift * guardias
    diesel_saved_usd_shift = diesel_saved_l_shift * float(g.precio_diesel_usd_l)
    diesel_saved_usd_year = diesel_saved_usd_shift * guardias
    co2_avoided_kg_shift = diesel_saved_l_shift * float(g.factor_co2_kg_l)
    co2_avoided_t_year = (co2_avoided_kg_shift * guardias) / 1000.0

    return {
        "n_trucks": n_trucks,
        "n_scoops_total": total_scoops,
        "dm_volquete": round(dm_volq, 4),
        "truck_fu": round(truck_fu, 4),
        # Ventilación D.S. 023-2017-EM (pico simultáneo interior mina por zona, Fu=1.0, sin Dm)
        "underground_trucks_peak_by_zone": underground_trucks_peak_by_zone,
        "underground_trucks_avg_by_zone": underground_trucks_avg_by_zone,
        "vent_q_peak_m3_min_by_zone": vent_q_peak_m3_min_by_zone,
        "vent_q_peak_cfm_by_zone": vent_q_peak_cfm_by_zone,
        "vent_q_peak_total_m3_min": vent_q_peak_total_m3_min,
        "vent_q_peak_total_cfm": vent_q_peak_total_cfm,
        "vent_cap_max_m3_min_by_zone": vent_cap_max_m3_min_by_zone,
        "vent_cap_usage_pct_by_zone": vent_cap_usage_pct_by_zone,
        "vent_margin_m3_min_by_zone": vent_margin_m3_min_by_zone,
        "vent_alert_by_zone": vent_alert_by_zone,
        "vent_any_alert": vent_any_alert,
        # Trío operativo-ambiental de volquetes en cola (guardia y año)
        "wait_hr_shift": round(wait_hr, 4),
        "diesel_wasted_wait_l_shift": round(diesel_wasted_wait_l_shift, 2),
        "diesel_wasted_wait_l_year": round(diesel_wasted_wait_l_year, 1),
        "diesel_wasted_wait_cost_usd_shift": round(diesel_wasted_wait_cost_usd_shift, 2),
        "diesel_wasted_wait_cost_usd_year": round(diesel_wasted_wait_cost_usd_year, 2),
        "co2_wasted_wait_kg_shift": round(co2_wasted_wait_kg_shift, 2),
        "co2_wasted_wait_t_year": round(co2_wasted_wait_t_year, 3),
        # Ahorro / CO2 evitado frente a escenario base
        "wait_saved_hr_shift": round(wait_saved_hr, 4),
        "diesel_saved_l_shift": round(diesel_saved_l_shift, 2),
        "diesel_saved_l_year": round(diesel_saved_l_year, 1),
        "diesel_saved_usd_shift": round(diesel_saved_usd_shift, 2),
        "diesel_saved_usd_year": round(diesel_saved_usd_year, 2),
        "co2_avoided_kg_shift": round(co2_avoided_kg_shift, 2),
        "co2_avoided_t_year": round(co2_avoided_t_year, 3),
    }


# ==============================================================================
# 3. EJECUTOR DE RÉPLICAS MONTECARLO (CON IC 95% Y CUOTAS POR ZONA)
# ==============================================================================
def run_replications(
    config: SimulationConfig,
    n_replicas: Optional[int] = None,
    base_seed: Optional[int] = None,
    seed_step: int = 1,
    confidence: Optional[float] = None,
) -> Dict[str, object]:
    """
    Ejecuta n_replicas corridas independientes con semillas deterministas
    `seed_i = base_seed + i * seed_step` y calcula IC t-Student, cuotas y ABET.
    Si `base_seed` es None, hereda `config.global_cfg.semilla`.
    """
    cfg_run = config.clone()
    g = cfg_run.global_cfg
    g.avoid_down_zones = True
    pol = normalize_policy(g.politica_despacho)
    g.politica_despacho = pol

    seed0 = int(g.semilla if base_seed is None else base_seed)
    n_rep = max(1, int(g.num_replicas if n_replicas is None else n_replicas))
    conf = float(g.nivel_confianza if confidence is None else confidence)
    active_zones = [z for z in cfg_run.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]

    # Sincronización de meta diaria como suma de cuotas de zonas activas
    active_quota_sum = sum(float(z.quota_mineral_tn) for z in active_zones)
    if active_quota_sum > 0.0:
        g.meta_mineral_gdia = active_quota_sum

    replica_rows: List[Dict[str, object]] = []
    raw_kpis: List[Dict[str, object]] = []

    for i in range(n_rep):
        sd = seed0 + i * seed_step
        kpi = SubHaulDESEngine(cfg_run, seed_override=sd).run()
        raw_kpis.append(kpi)

        eff_horizon_sec = float(kpi["effective_horizon_sec"])
        row: Dict[str, object] = {
            "replica": i + 1,
            "seed": sd,
            "effective_horizon_sec": eff_horizon_sec,
            "effective_horizon_hr": eff_horizon_sec / 3600.0,
            "ore_total_tn": float(kpi["ore_total_tn"]),
            "capped_ore_tn": float(kpi["capped_ore_tn"]),
            "waste_total_tn": float(kpi["waste_total_tn"]),
            "gross_revenue_usd": float(kpi["gross_revenue_usd"]),
            "scoop_op_cost_usd": float(kpi["scoop_op_cost_usd"]),
            "scoop_idle_cost_usd": float(kpi["scoop_idle_cost_usd"]),
            "truck_op_cost_usd": float(kpi["truck_op_cost_usd"]),
            "truck_wait_cost_usd": float(kpi["truck_wait_cost_usd"]),
            "fleet_op_cost_usd": float(kpi["fleet_op_cost_usd"]),
            "total_opex_usd": float(kpi["total_opex_usd"]),
            "net_revenue_usd": float(kpi["net_revenue_usd"]),
            "unit_cost_usd_tn": float(kpi["unit_cost_usd_tn"]),
            "total_truck_wait_hr": float(kpi["total_truck_wait_hr"]),
            "scale_wait_hr": float(kpi["scale_wait_hr"]),
            "scale_utilization": float(kpi["scale_utilization"]),
            "truck_fu": float(kpi["truck_fu"]),
            "target_met": bool(kpi["target_met"]),
            "time_to_target_sec": kpi.get("time_to_target_sec"),
            "time_to_target_hr": kpi.get("time_to_target_hr"),
            "vent_q_peak_total_m3_min": float(kpi["vent_q_peak_total_m3_min"]),
            "vent_q_peak_total_cfm": float(kpi["vent_q_peak_total_cfm"]),
            "vent_any_alert": bool(kpi["vent_any_alert"]),
            "diesel_wasted_wait_l_shift": float(kpi["diesel_wasted_wait_l_shift"]),
            "diesel_wasted_wait_l_year": float(kpi["diesel_wasted_wait_l_year"]),
            "diesel_wasted_wait_cost_usd_shift": float(kpi["diesel_wasted_wait_cost_usd_shift"]),
            "diesel_wasted_wait_cost_usd_year": float(kpi["diesel_wasted_wait_cost_usd_year"]),
            "co2_wasted_wait_kg_shift": float(kpi["co2_wasted_wait_kg_shift"]),
            "co2_wasted_wait_t_year": float(kpi["co2_wasted_wait_t_year"]),
        }
        for zid in zone_ids:
            row[f"ore_{zid}"] = kpi["ore_by_zone"][zid]
            row[f"util_{zid}"] = kpi["scoop_utilization_by_zone"][zid]
            row[f"wait_hr_{zid}"] = kpi["wait_hr_by_zone"][zid]
            row[f"max_q_{zid}"] = kpi["max_queue_by_station"][zid]
            row[f"peak_int_{zid}"] = int(kpi["underground_trucks_peak_by_zone"][zid])
            row[f"avg_int_{zid}"] = float(kpi["underground_trucks_avg_by_zone"][zid])
            row[f"vent_q_peak_m3_{zid}"] = float(kpi["vent_q_peak_m3_min_by_zone"][zid])
            row[f"vent_q_peak_cfm_{zid}"] = float(kpi["vent_q_peak_cfm_by_zone"][zid])
        replica_rows.append(row)

    df_rep = pd.DataFrame(replica_rows)

    ore_ci = compute_mean_ci(df_rep["ore_total_tn"], conf)
    waste_ci = compute_mean_ci(df_rep["waste_total_tn"], conf)
    total_opex_ci = compute_mean_ci(df_rep["total_opex_usd"], conf)
    fleet_op_cost_ci = compute_mean_ci(df_rep["fleet_op_cost_usd"], conf)
    net_ci = compute_mean_ci(df_rep["net_revenue_usd"], conf)
    unit_cost_ci = compute_mean_ci(df_rep["unit_cost_usd_tn"], conf)
    wait_ci = compute_mean_ci(df_rep["total_truck_wait_hr"], conf)
    fu_ci = compute_mean_ci(df_rep["truck_fu"], conf)
    horizon_hr_ci = compute_mean_ci(df_rep["effective_horizon_hr"], conf)
    vent_q_total_m3_ci = compute_mean_ci(df_rep["vent_q_peak_total_m3_min"], conf)
    vent_q_total_cfm_ci = compute_mean_ci(df_rep["vent_q_peak_total_cfm"], conf)
    diesel_l_shift_ci = compute_mean_ci(df_rep["diesel_wasted_wait_l_shift"], conf)
    diesel_usd_shift_ci = compute_mean_ci(df_rep["diesel_wasted_wait_cost_usd_shift"], conf)
    co2_kg_shift_ci = compute_mean_ci(df_rep["co2_wasted_wait_kg_shift"], conf)

    valid_ttt = df_rep["time_to_target_hr"].dropna().astype(float).tolist()
    time_to_target_hr_ci = compute_mean_ci(valid_ttt, conf) if len(valid_ttt) > 0 else None

    ore_zone_means = {zid: round(float(df_rep[f"ore_{zid}"].mean()), 2) for zid in zone_ids}
    util_zone_means = {zid: round(float(df_rep[f"util_{zid}"].mean()), 4) for zid in zone_ids}
    max_q_zone_means = {zid: round(float(df_rep[f"max_q_{zid}"].mean()), 2) for zid in zone_ids}

    # Para ventilación D.S. 023-2017-EM se conserva el pico máximo simultáneo entero observado
    # por zona (sin promediar volquetes pico); el promedio interior mina queda solo como dato informativo.
    underground_peak_by_zone = {zid: int(df_rep[f"peak_int_{zid}"].max()) for zid in zone_ids}
    underground_avg_by_zone_mean = {zid: round(float(df_rep[f"avg_int_{zid}"].mean()), 4) for zid in zone_ids}

    quota_ratios = []
    quota_compliance_by_zone: Dict[str, float] = {}
    for z in active_zones:
        if z.quota_mineral_tn > 0:
            pct = (ore_zone_means[z.zone_id] / z.quota_mineral_tn) * 100.0
            quota_ratios.append(min(1.0, pct / 100.0))
            quota_compliance_by_zone[z.zone_id] = round(pct, 2)
        else:
            quota_compliance_by_zone[z.zone_id] = 100.0

    avg_quota_pct = round(float(np.mean(quota_ratios) * 100.0), 2) if quota_ratios else 100.0
    min_quota_pct = round(float(min(quota_ratios) * 100.0), 2) if quota_ratios else 100.0

    abet = compute_abet_metrics(
        cfg_run,
        {
            "n_trucks_total": int(raw_kpis[0]["n_trucks_total"]),
            "truck_fu": fu_ci["mean"],
            "total_truck_wait_hr": wait_ci["mean"],
            "underground_trucks_peak_by_zone": underground_peak_by_zone,
            "underground_trucks_avg_by_zone": underground_avg_by_zone_mean,
        },
    )

    return {
        "base_seed": seed0,
        "n_replicas": n_rep,
        "confidence": conf,
        "policy": pol,
        "policy_label": POLICY_LABELS.get(pol, pol),
        "stop_mode": g.modo_parada,
        "meta_mineral_gdia": float(g.meta_mineral_gdia),
        "effective_horizon_hr": horizon_hr_ci,
        "n_trucks_total": int(raw_kpis[0]["n_trucks_total"]),
        "dm_volquete": round(float(g.dm_volquete), 4),
        "assignment": dict(raw_kpis[0]["assignment"]),
        "assignment_str": format_assignment(raw_kpis[0]["assignment"], zone_ids),
        "ore_tn": ore_ci,
        "waste_tn": waste_ci,
        "total_opex_usd": total_opex_ci,
        "fleet_op_cost_usd": fleet_op_cost_ci,
        "net_revenue_usd": net_ci,
        "unit_cost_usd_tn": unit_cost_ci,
        "truck_wait_hr": wait_ci,
        "truck_fu": fu_ci,
        "vent_q_peak_total_m3_min_ci": vent_q_total_m3_ci,
        "vent_q_peak_total_cfm_ci": vent_q_total_cfm_ci,
        "diesel_wasted_wait_l_shift_ci": diesel_l_shift_ci,
        "diesel_wasted_wait_cost_usd_shift_ci": diesel_usd_shift_ci,
        "co2_wasted_wait_kg_shift_ci": co2_kg_shift_ci,
        "prob_target_met_pct": round(float(df_rep["target_met"].mean() * 100.0), 2),
        "target_met_mean": ore_ci["mean"] >= g.meta_mineral_gdia,
        "time_to_target_hr": time_to_target_hr_ci,
        "n_replicas_target_met": len(valid_ttt),
        "ore_by_zone_mean": ore_zone_means,
        "scoop_util_by_zone_mean": util_zone_means,
        "max_queue_by_zone_mean": max_q_zone_means,
        "underground_trucks_peak_by_zone": underground_peak_by_zone,
        "underground_trucks_avg_by_zone_mean": underground_avg_by_zone_mean,
        "quota_compliance_by_zone_pct": quota_compliance_by_zone,
        "avg_quota_compliance_pct": avg_quota_pct,
        "min_quota_compliance_pct": min_quota_pct,
        "abet": abet,
        "replicas_df": df_rep,
    }


# ==============================================================================
# 4. CONTRASTE ESTADÍSTICO FORMAL (1 vs 1) Y BENCHMARK SIMULTÁNEO (CRN)
# ==============================================================================
def _safe_hypothesis_tests(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, object]:
    """Contrasta B vs A (diff = B - A) con pruebas paramétricas y no paramétricas."""
    diff = sample_b - sample_a
    diff_ci = compute_mean_ci(diff, confidence)
    alpha = 1.0 - confidence

    if len(diff) < 2 or np.allclose(diff, 0.0):
        return {
            "delta_mean": diff_ci["mean"],
            "delta_ci_low": diff_ci["ci_low"],
            "delta_ci_high": diff_ci["ci_high"],
            "t_paired_stat": 0.0,
            "t_paired_p": 1.0,
            "t_welch_stat": 0.0,
            "t_welch_p": 1.0,
            "mannwhitney_stat": 0.0,
            "mannwhitney_p": 1.0,
            "wilcoxon_stat": 0.0,
            "wilcoxon_p": 1.0,
            "significant_parametric": False,
            "significant_nonparametric": False,
            "alpha": alpha,
        }

    t_pair_stat, t_pair_p = stats.ttest_rel(sample_b, sample_a)
    t_welch_stat, t_welch_p = stats.ttest_ind(sample_b, sample_a, equal_var=False)
    mwu_stat, mwu_p = stats.mannwhitneyu(sample_b, sample_a, alternative="two-sided")
    try:
        wilc_stat, wilc_p = stats.wilcoxon(sample_b, sample_a, alternative="two-sided")
    except ValueError:
        wilc_stat, wilc_p = 0.0, 1.0

    t_pair_p = 1.0 if math.isnan(t_pair_p) else float(t_pair_p)
    t_welch_p = 1.0 if math.isnan(t_welch_p) else float(t_welch_p)
    mwu_p = 1.0 if math.isnan(mwu_p) else float(mwu_p)
    wilc_p = 1.0 if math.isnan(wilc_p) else float(wilc_p)

    return {
        "delta_mean": round(diff_ci["mean"], 4),
        "delta_ci_low": round(diff_ci["ci_low"], 4),
        "delta_ci_high": round(diff_ci["ci_high"], 4),
        "t_paired_stat": round(float(t_pair_stat), 4),
        "t_paired_p": t_pair_p,
        "t_welch_stat": round(float(t_welch_stat), 4),
        "t_welch_p": t_welch_p,
        "mannwhitney_stat": round(float(mwu_stat), 4),
        "mannwhitney_p": mwu_p,
        "wilcoxon_stat": round(float(wilc_stat), 4),
        "wilcoxon_p": wilc_p,
        "significant_parametric": t_pair_p < alpha,
        "significant_nonparametric": mwu_p < alpha,
        "alpha": alpha,
    }


def compare_policies_statistically(
    config: SimulationConfig,
    policy_base: str = "FIJA",
    policy_alt: str = "DIN_ETA_RITMO",
    n_replicas: Optional[int] = None,
    base_seed: Optional[int] = None,
    assignment_base: Optional[Dict[str, int]] = None,
    assignment_alt: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Compara dos configuraciones usando Common Random Numbers (CRN)."""
    pol_a = normalize_policy(policy_base)
    pol_b = normalize_policy(policy_alt)
    seed0 = int(config.global_cfg.semilla if base_seed is None else base_seed)

    cfg_a = config.clone()
    cfg_a.global_cfg.avoid_down_zones = True
    cfg_a.global_cfg.politica_despacho = pol_a
    if assignment_base is not None:
        cfg_a = cfg_a.set_fleet_assignment(assignment_base)

    cfg_b = config.clone()
    cfg_b.global_cfg.avoid_down_zones = True
    cfg_b.global_cfg.politica_despacho = pol_b
    if assignment_alt is not None:
        cfg_b = cfg_b.set_fleet_assignment(assignment_alt)

    res_a = run_replications(cfg_a, n_replicas=n_replicas, base_seed=seed0)
    res_b = run_replications(cfg_b, n_replicas=n_replicas, base_seed=seed0)

    df_a, df_b = res_a["replicas_df"], res_b["replicas_df"]
    conf = config.global_cfg.nivel_confianza
    guardias = int(config.global_cfg.guardias_anio)

    test_ore = _safe_hypothesis_tests(df_a["ore_total_tn"].values, df_b["ore_total_tn"].values, conf)
    test_net = _safe_hypothesis_tests(df_a["net_revenue_usd"].values, df_b["net_revenue_usd"].values, conf)
    test_wait = _safe_hypothesis_tests(df_a["total_truck_wait_hr"].values, df_b["total_truck_wait_hr"].values, conf)
    test_vent = _safe_hypothesis_tests(
        df_a["vent_q_peak_total_m3_min"].values, df_b["vent_q_peak_total_m3_min"].values, conf
    )
    test_diesel = _safe_hypothesis_tests(
        df_a["diesel_wasted_wait_l_shift"].values, df_b["diesel_wasted_wait_l_shift"].values, conf
    )
    test_co2 = _safe_hypothesis_tests(
        df_a["co2_wasted_wait_kg_shift"].values, df_b["co2_wasted_wait_kg_shift"].values, conf
    )

    abet_b_vs_a = compute_abet_metrics(
        cfg_b,
        {
            "n_trucks_total": res_b["n_trucks_total"],
            "truck_fu": res_b["truck_fu"]["mean"],
            "total_truck_wait_hr": res_b["truck_wait_hr"]["mean"],
            "underground_trucks_peak_by_zone": res_b["underground_trucks_peak_by_zone"],
            "underground_trucks_avg_by_zone": res_b["underground_trucks_avg_by_zone_mean"],
        },
        baseline_wait_hr=res_a["truck_wait_hr"]["mean"],
    )

    delta_net_gdia = float(test_net["delta_mean"])
    delta_net_anio = round(delta_net_gdia * guardias, 2)

    lbl_a = f"Base ({POLICY_LABELS.get(pol_a, pol_a)} | {res_a['assignment_str']})"
    lbl_b = f"Alternativa ({POLICY_LABELS.get(pol_b, pol_b)} | {res_b['assignment_str']})"

    summary_rows = [
        {
            "Métrica": "Producción Mineral (t/gdia)",
            lbl_a: f"{res_a['ore_tn']['mean']:.2f} ± {res_a['ore_tn']['ci_half']:.2f}",
            lbl_b: f"{res_b['ore_tn']['mean']:.2f} ± {res_b['ore_tn']['ci_half']:.2f}",
            "Delta (Alt - Base)": round(test_ore["delta_mean"], 2),
            "p-valor (t pareada)": round(test_ore["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_ore["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_ore["significant_parametric"] else "NO",
        },
        {
            "Métrica": "Ingreso Neto (US$/gdia)",
            lbl_a: f"{res_a['net_revenue_usd']['mean']:,.2f} ± {res_a['net_revenue_usd']['ci_half']:,.2f}",
            lbl_b: f"{res_b['net_revenue_usd']['mean']:,.2f} ± {res_b['net_revenue_usd']['ci_half']:,.2f}",
            "Delta (Alt - Base)": round(delta_net_gdia, 2),
            "p-valor (t pareada)": round(test_net["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_net["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_net["significant_parametric"] else "NO",
        },
        {
            "Métrica": "Tiempo en Cola Total (h/gdia)",
            lbl_a: f"{res_a['truck_wait_hr']['mean']:.3f} ± {res_a['truck_wait_hr']['ci_half']:.3f}",
            lbl_b: f"{res_b['truck_wait_hr']['mean']:.3f} ± {res_b['truck_wait_hr']['ci_half']:.3f}",
            "Delta (Alt - Base)": round(test_wait["delta_mean"], 3),
            "p-valor (t pareada)": round(test_wait["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_wait["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_wait["significant_parametric"] else "NO",
        },
        {
            "Métrica": "Caudal Pico Total (m3/min)",
            lbl_a: f"{res_a['vent_q_peak_total_m3_min_ci']['mean']:,.1f} ± {res_a['vent_q_peak_total_m3_min_ci']['ci_half']:,.1f}",
            lbl_b: f"{res_b['vent_q_peak_total_m3_min_ci']['mean']:,.1f} ± {res_b['vent_q_peak_total_m3_min_ci']['ci_half']:,.1f}",
            "Delta (Alt - Base)": round(test_vent["delta_mean"], 1),
            "p-valor (t pareada)": round(test_vent["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_vent["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_vent["significant_parametric"] else "NO",
        },
        {
            "Métrica": "Diésel Desperdiciado Colas (L/gdia)",
            lbl_a: f"{res_a['diesel_wasted_wait_l_shift_ci']['mean']:.2f} ± {res_a['diesel_wasted_wait_l_shift_ci']['ci_half']:.2f}",
            lbl_b: f"{res_b['diesel_wasted_wait_l_shift_ci']['mean']:.2f} ± {res_b['diesel_wasted_wait_l_shift_ci']['ci_half']:.2f}",
            "Delta (Alt - Base)": round(test_diesel["delta_mean"], 2),
            "p-valor (t pareada)": round(test_diesel["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_diesel["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_diesel["significant_parametric"] else "NO",
        },
        {
            "Métrica": "CO2 Colas Volquetes (kg/gdia)",
            lbl_a: f"{res_a['co2_wasted_wait_kg_shift_ci']['mean']:.2f} ± {res_a['co2_wasted_wait_kg_shift_ci']['ci_half']:.2f}",
            lbl_b: f"{res_b['co2_wasted_wait_kg_shift_ci']['mean']:.2f} ± {res_b['co2_wasted_wait_kg_shift_ci']['ci_half']:.2f}",
            "Delta (Alt - Base)": round(test_co2["delta_mean"], 2),
            "p-valor (t pareada)": round(test_co2["t_paired_p"], 5),
            "p-valor (Mann-Whitney U)": round(test_co2["mannwhitney_p"], 5),
            "Significativo (p<0.05)": "SÍ" if test_co2["significant_parametric"] else "NO",
        },
    ]

    return {
        "base_seed": seed0,
        "policy_base": pol_a,
        "policy_base_label": POLICY_LABELS.get(pol_a, pol_a),
        "policy_alt": pol_b,
        "policy_alt_label": POLICY_LABELS.get(pol_b, pol_b),
        "result_base": res_a,
        "result_alt": res_b,
        "test_ore": test_ore,
        "test_net": test_net,
        "test_wait": test_wait,
        "test_vent": test_vent,
        "test_diesel": test_diesel,
        "test_co2": test_co2,
        "delta_net_usd_gdia": round(delta_net_gdia, 2),
        "delta_net_usd_anio": delta_net_anio,
        "abet_comparison": abet_b_vs_a,
        "comparison_table": pd.DataFrame(summary_rows),
    }


def benchmark_all_policies(
    config: SimulationConfig,
    policies: Sequence[str] = ("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
    n_replicas: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Ejecuta en simultáneo todas las políticas indicadas usando Common Random Numbers (CRN)
    sobre la flota configurada y construye una matriz comparativa ejecutiva.
    """
    seed0 = int(config.global_cfg.semilla if base_seed is None else base_seed)
    active_zones = [z for z in config.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]

    results_by_policy: Dict[str, Dict[str, object]] = {}
    for pol in policies:
        pol_n = normalize_policy(pol)
        cfg_p = config.clone()
        cfg_p.global_cfg.avoid_down_zones = True
        cfg_p.global_cfg.politica_despacho = pol_n
        results_by_policy[pol_n] = run_replications(cfg_p, n_replicas=n_replicas, base_seed=seed0)

    # Referencia base para calcular deltas y ahorro de combustible/CO2
    base_key = "FIJA" if "FIJA" in results_by_policy else next(iter(results_by_policy))
    base_res = results_by_policy[base_key]
    base_wait = base_res["truck_wait_hr"]["mean"]
    base_net = base_res["net_revenue_usd"]["mean"]

    rows: List[Dict[str, object]] = []
    for pol_n, res in results_by_policy.items():
        wait_mean = res["truck_wait_hr"]["mean"]
        net_mean = res["net_revenue_usd"]["mean"]
        delta_wait_pct = ((wait_mean - base_wait) / base_wait * 100.0) if base_wait > 0 else 0.0

        cfg_p = config.clone()
        cfg_p.global_cfg.avoid_down_zones = True
        cfg_p.global_cfg.politica_despacho = pol_n
        abet_vs_base = compute_abet_metrics(
            cfg_p,
            {
                "n_trucks_total": res["n_trucks_total"],
                "truck_fu": res["truck_fu"]["mean"],
                "total_truck_wait_hr": wait_mean,
                "underground_trucks_peak_by_zone": res["underground_trucks_peak_by_zone"],
                "underground_trucks_avg_by_zone": res["underground_trucks_avg_by_zone_mean"],
            },
            baseline_wait_hr=base_wait,
        )

        row: Dict[str, object] = {
            "Código": pol_n,
            "Política": POLICY_LABELS.get(pol_n, pol_n),
            "Mineral (t/gdia)": round(res["ore_tn"]["mean"], 2),
            "IC95 Mineral (±)": round(res["ore_tn"]["ci_half"], 2),
            "Ingreso Neto (US$/gdia)": round(net_mean, 2),
            "OPEX Total (US$/gdia)": round(res["total_opex_usd"]["mean"], 2),
            "Costo Flota Volq. (US$/gdia)": round(res["fleet_op_cost_usd"]["mean"], 2),
            "Δ Neto vs Fija (US$/gdia)": round(net_mean - base_net, 2),
            "Costo Unitario (US$/t)": round(res["unit_cost_usd_tn"]["mean"], 3),
            "Cola Total (h/gdia)": round(wait_mean, 3),
            "Δ Cola vs Fija (%)": round(delta_wait_pct, 1),
            "Cumpl. Mín. Cuota (%)": res["min_quota_compliance_pct"],
            "Ventilación Q Pico (m3/min)": abet_vs_base["vent_q_peak_total_m3_min"],
            "Ventilación Q Pico (CFM)": abet_vs_base["vent_q_peak_total_cfm"],
            "Diésel Cola (L/gdia)": abet_vs_base["diesel_wasted_wait_l_shift"],
            "Costo Diésel Cola (US$/gdia)": abet_vs_base["diesel_wasted_wait_cost_usd_shift"],
            "CO2 Cola (kg/gdia)": abet_vs_base["co2_wasted_wait_kg_shift"],
            "CO2 Cola (t/año)": abet_vs_base["co2_wasted_wait_t_year"],
            "CO2 Evitado vs Fija (t/año)": abet_vs_base["co2_avoided_t_year"],
        }
        for zid in zone_ids:
            row[f"Ore_{zid} (t)"] = res["ore_by_zone_mean"][zid]
            row[f"Q_pico_{zid} (m3/min)"] = abet_vs_base["vent_q_peak_m3_min_by_zone"][zid]
        rows.append(row)

    return {
        "base_seed": seed0,
        "base_policy": base_key,
        "summary_df": pd.DataFrame(rows),
        "results_by_policy": results_by_policy,
    }


# ==============================================================================
# 5. OPTIMIZADOR PRESCRIPTIVO (SIM-OPT) PARA FLOTA Y ASIGNACIÓN EN N ZONAS
# ==============================================================================
def _generate_compositions(n_total: int, n_zones: int, min_per_zone: int = 1) -> List[Tuple[int, ...]]:
    """Genera todas las tuplas (n_1, ..., n_Z) con sum = n_total y n_i >= min_per_zone."""
    if n_zones <= 0:
        return []
    if n_zones == 1:
        return [(n_total,)]
    eff_min = min_per_zone if n_total >= n_zones * min_per_zone else 0
    rem = n_total - n_zones * eff_min
    result: List[Tuple[int, ...]] = []
    for bars in itertools.combinations(range(rem + n_zones - 1), n_zones - 1):
        parts = []
        prev = -1
        for b in bars:
            parts.append((b - prev - 1) + eff_min)
            prev = b
        parts.append((rem + n_zones - 1 - prev - 1) + eff_min)
        result.append(tuple(parts))
    return result


def _initial_proportional_assignment(config: SimulationConfig, n_total: int, min_per_zone: int = 1) -> Tuple[int, ...]:
    """Semilla inicial proporcional a cuotas de zona (para Z grande)."""
    active_zones = [z for z in config.zones if z.active]
    k = len(active_zones)
    eff_min = min_per_zone if n_total >= k * min_per_zone else 0
    weights = np.array([z.quota_mineral_tn if z.quota_mineral_tn > 0 else 1.0 for z in active_zones], dtype=float)
    weights = weights / weights.sum()

    alloc = np.full(k, eff_min, dtype=int)
    rem = n_total - int(alloc.sum())
    if rem > 0:
        raw = weights * rem
        floors = np.floor(raw).astype(int)
        alloc += floors
        leftover = rem - int(floors.sum())
        if leftover > 0:
            for idx in np.argsort(-(raw - floors))[:leftover]:
                alloc[idx] += 1
    return tuple(int(x) for x in alloc)


def _objective_score(
    res: Dict[str, object],
    objective: str,
    meta_mineral: float,
    require_target_met: bool,
    min_zone_quota_pct: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Tupla ordenable con progreso continuo hacia factibilidad (elimina mesetas 0/1):
    (progreso_cuotas_zonas, progreso_meta_global, valor_objetivo).
    """
    obj = objective.strip().upper()
    quota_prog = (
        min(1.0, res["min_quota_compliance_pct"] / min_zone_quota_pct)
        if min_zone_quota_pct > 0.0
        else 1.0
    )
    target_prog = (
        min(1.0, res["ore_tn"]["mean"] / meta_mineral)
        if (require_target_met and meta_mineral > 0.0)
        else 1.0
    )
    if obj == "TONELAJE":
        val = float(res["ore_tn"]["mean"])
    elif obj == "COSTO_UNITARIO":
        val = -float(res["unit_cost_usd_tn"]["mean"])
    else:
        val = float(res["net_revenue_usd"]["mean"])
    return (round(quota_prog, 4), round(target_prog, 4), val)


def optimize_fleet_and_assignment(
    config: SimulationConfig,
    fleet_min: Optional[int] = None,
    fleet_max: Optional[int] = None,
    policy: Optional[str] = None,
    objective: Optional[str] = None,
    n_replicas_screen: int = 15,
    n_replicas_confirm: int = 30,
    top_k_confirm: int = 3,
    min_trucks_per_zone: int = 1,
    require_target_met: bool = True,
    min_zone_quota_pct: float = 0.0,
    base_seed: Optional[int] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    use_plant_cap: bool = False,
    plant_cap_tn: float = 0.0,
) -> Dict[str, object]:
    """
    Optimizador Sim-Opt en dos etapas (screening rápido + confirmación estadística)
    para cualquier topología de N zonas.
    """
    cfg_work = config.clone()
    g = cfg_work.global_cfg
    g.avoid_down_zones = True

    active_zones = [z for z in cfg_work.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]
    n_zones = len(zone_ids)

    # Sincronización de meta diaria como suma de cuotas de zonas activas
    active_quota_sum = sum(float(z.quota_mineral_tn) for z in active_zones)
    if active_quota_sum > 0.0:
        g.meta_mineral_gdia = active_quota_sum

    g.usar_tope_planta = bool(use_plant_cap)
    g.cap_planta_tn = float(plant_cap_tn) if use_plant_cap else 0.0
    if policy is not None:
        g.politica_despacho = normalize_policy(policy)
    else:
        g.politica_despacho = normalize_policy(g.politica_despacho)
    obj = (g.funcion_objetivo if objective is None else objective).strip().upper()
    seed0 = int(g.semilla if base_seed is None else base_seed)

    # Límite de referencia para calcular el excedente sobre el tope/meta
    limite_ref = (
        float(plant_cap_tn)
        if (use_plant_cap and float(plant_cap_tn) > 0.0)
        else float(g.meta_mineral_gdia)
    )

    f_min = int(g.flota_min if fleet_min is None else fleet_min)
    f_max = int(g.flota_max if fleet_max is None else fleet_max)
    if f_max < f_min:
        f_min, f_max = f_max, f_min

    fleet_sizes = list(range(f_min, f_max + 1))

    all_confirmed_records: List[Dict[str, object]] = []
    best_per_fleet_records: List[Dict[str, object]] = []
    best_per_fleet_full: Dict[int, Dict[str, object]] = {}

    for step_idx, n_fleet in enumerate(fleet_sizes):
        if progress_callback:
            progress_callback(step_idx / max(1, len(fleet_sizes)), f"Optimizando flota N = {n_fleet} volquetes...")

        eff_min = min_trucks_per_zone if n_fleet >= n_zones * min_trucks_per_zone else 0
        screen_cache: Dict[Tuple[int, ...], Dict[str, object]] = {}

        def eval_screen(tup: Tuple[int, ...]) -> Dict[str, object]:
            if tup not in screen_cache:
                asig_dict = {zid: int(cnt) for zid, cnt in zip(zone_ids, tup)}
                screen_cache[tup] = run_replications(
                    cfg_work.set_fleet_assignment(asig_dict),
                    n_replicas=n_replicas_screen,
                    base_seed=seed0,
                )
            return screen_cache[tup]

        for tup in _generate_compositions(n_fleet, n_zones, min_per_zone=eff_min):
            eval_screen(tup)

        ranked_screen = sorted(
            screen_cache.items(),
            key=lambda kv: _objective_score(kv[1], obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct),
            reverse=True,
        )
        top_tuples = [tup for tup, _ in ranked_screen[: max(1, min(top_k_confirm, len(ranked_screen)))]]

        confirmed_for_fleet: List[Tuple[Tuple[int, ...], Dict[str, object]]] = []
        for tup in top_tuples:
            if n_replicas_confirm == n_replicas_screen:
                res_conf = screen_cache[tup]
            else:
                asig_dict = {zid: int(cnt) for zid, cnt in zip(zone_ids, tup)}
                res_conf = run_replications(
                    cfg_work.set_fleet_assignment(asig_dict),
                    n_replicas=n_replicas_confirm,
                    base_seed=seed0,
                )
            confirmed_for_fleet.append((tup, res_conf))

            abet_c = res_conf["abet"]
            ore_mean_val = float(res_conf["ore_tn"]["mean"])
            excedente_tope = round(max(0.0, ore_mean_val - limite_ref), 2)

            ttt_ci = res_conf.get("time_to_target_hr")
            if ttt_ci is not None:
                ttt_mean_hr: Optional[float] = round(float(ttt_ci["mean"]), 2)
                ttt_str = _format_hours_hm(float(ttt_ci["mean"]))
            else:
                ttt_mean_hr = None
                ttt_str = "No alcanzada"

            rec: Dict[str, object] = {
                "Flota (N)": n_fleet,
                "Asignación": res_conf["assignment_str"],
                "Política": POLICY_LABELS.get(g.politica_despacho, g.politica_despacho),
                "Mineral (t/gdia)": round(ore_mean_val, 2),
                "IC95 Mineral (±)": round(res_conf["ore_tn"]["ci_half"], 2),
                "Excedente Tope (t)": excedente_tope,
                "Cumple Meta (%)": res_conf["prob_target_met_pct"],
                "Tiempo a Meta": ttt_str,
                "Tiempo a Meta (h)": ttt_mean_hr,
                "Cumpl. Mín. Cuota Zona (%)": res_conf["min_quota_compliance_pct"],
                "Ingreso Neto (US$/gdia)": round(res_conf["net_revenue_usd"]["mean"], 2),
                "IC95 Neto (±)": round(res_conf["net_revenue_usd"]["ci_half"], 2),
                "OPEX Total (US$/gdia)": round(res_conf["total_opex_usd"]["mean"], 2),
                "Costo Flota Volq. (US$/gdia)": round(res_conf["fleet_op_cost_usd"]["mean"], 2),
                "Costo Unitario (US$/t)": round(res_conf["unit_cost_usd_tn"]["mean"], 3),
                "Cola Total (h/gdia)": round(res_conf["truck_wait_hr"]["mean"], 3),
                "Fu Volquetes": round(res_conf["truck_fu"]["mean"], 4),
                "Ventilación Q Pico (m3/min)": abet_c["vent_q_peak_total_m3_min"],
                "Ventilación Q Pico (CFM)": abet_c["vent_q_peak_total_cfm"],
                "Alerta Vent.": "SÍ" if abet_c["vent_any_alert"] else "NO",
                "Diésel Cola (L/gdia)": abet_c["diesel_wasted_wait_l_shift"],
                "Costo Diésel Cola (US$/gdia)": abet_c["diesel_wasted_wait_cost_usd_shift"],
                "CO2 Cola (kg/gdia)": abet_c["co2_wasted_wait_kg_shift"],
                "CO2 Cola (t/año)": abet_c["co2_wasted_wait_t_year"],
            }
            for zid in zone_ids:
                rec[f"Ore_{zid} (t)"] = res_conf["ore_by_zone_mean"][zid]
                rec[f"Q_pico_{zid} (m3/min)"] = abet_c["vent_q_peak_m3_min_by_zone"][zid]
            all_confirmed_records.append(rec)

        _, best_res = max(
            confirmed_for_fleet,
            key=lambda kv: _objective_score(kv[1], obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct),
        )
        best_per_fleet_full[n_fleet] = best_res
        best_row = next(
            r for r in reversed(all_confirmed_records)
            if r["Flota (N)"] == n_fleet and r["Asignación"] == best_res["assignment_str"]
        )
        best_per_fleet_records.append(dict(best_row))

    if progress_callback:
        progress_callback(1.0, "Optimización Sim-Opt completada.")

    df_curve = pd.DataFrame(best_per_fleet_records).sort_values("Flota (N)").reset_index(drop=True)
    df_curve["Δ Mineral/Volq (t)"] = df_curve["Mineral (t/gdia)"].diff().round(2).fillna(0.0)
    df_curve["Δ Ingreso Neto/Volq (US$/gdia)"] = df_curve["Ingreso Neto (US$/gdia)"].diff().round(2).fillna(0.0)
    df_curve["Δ Ventilación/Volq (m3/min)"] = df_curve["Ventilación Q Pico (m3/min)"].diff().round(2).fillna(0.0)

    best_fleet_n, best_overall = max(
        best_per_fleet_full.items(),
        key=lambda kv: _objective_score(kv[1], obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct),
    )

    feasible_fleets = [
        n for n, res in sorted(best_per_fleet_full.items())
        if res["ore_tn"]["mean"] >= g.meta_mineral_gdia
        and (min_zone_quota_pct <= 0.0 or res["min_quota_compliance_pct"] >= min_zone_quota_pct)
    ]
    min_fleet = feasible_fleets[0] if feasible_fleets else None

    sort_col = (
        "Ingreso Neto (US$/gdia)" if obj == "INGRESO_NETO"
        else ("Mineral (t/gdia)" if obj == "TONELAJE" else "Costo Unitario (US$/t)")
    )
    df_all = pd.DataFrame(all_confirmed_records).sort_values(
        by=sort_col, ascending=(obj == "COSTO_UNITARIO")
    ).reset_index(drop=True)

    return {
        "base_seed": seed0,
        "objective": obj,
        "policy": g.politica_despacho,
        "policy_label": POLICY_LABELS.get(g.politica_despacho, g.politica_despacho),
        "meta_mineral_gdia": float(g.meta_mineral_gdia),
        "use_plant_cap": bool(use_plant_cap),
        "plant_cap_tn": float(plant_cap_tn),
        "limite_ref_tn": float(limite_ref),
        "best_fleet_n": best_fleet_n,
        "best_assignment": best_overall["assignment"],
        "best_assignment_str": best_overall["assignment_str"],
        "best_overall": best_overall,
        "min_fleet_meeting_target": min_fleet,
        "min_fleet_result": best_per_fleet_full[min_fleet] if min_fleet is not None else None,
        "fleet_curve_df": df_curve,
        "all_candidates_df": df_all,
    }


def compare_optimal_fixed_vs_dynamic(
    config: SimulationConfig,
    fleet_size: int,
    policy_dynamic: str = "DIN_ETA_RITMO",
    objective: str = "INGRESO_NETO",
    n_replicas_screen: int = 15,
    n_replicas_confirm: int = 30,
    min_zone_quota_pct: float = 90.0,
    base_seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Contraste metodológicamente justo:
    Optimiza por separado el Mejor Escenario Fijo y el Mejor Escenario Dinámico
    para la misma flota N y restricciones, y luego corre las pruebas formales
    (t-Student y Mann-Whitney U) entre ambos ganadores.
    """
    seed0 = int(config.global_cfg.semilla if base_seed is None else base_seed)
    pol_dyn = normalize_policy(policy_dynamic)
    opt_fixed = optimize_fleet_and_assignment(
        config,
        fleet_min=fleet_size,
        fleet_max=fleet_size,
        policy="FIJA",
        objective=objective,
        n_replicas_screen=n_replicas_screen,
        n_replicas_confirm=n_replicas_confirm,
        top_k_confirm=3,
        require_target_met=True,
        min_zone_quota_pct=min_zone_quota_pct,
        base_seed=seed0,
    )
    opt_dyn = optimize_fleet_and_assignment(
        config,
        fleet_min=fleet_size,
        fleet_max=fleet_size,
        policy=pol_dyn,
        objective=objective,
        n_replicas_screen=n_replicas_screen,
        n_replicas_confirm=n_replicas_confirm,
        top_k_confirm=3,
        require_target_met=True,
        min_zone_quota_pct=min_zone_quota_pct,
        base_seed=seed0,
    )

    comp = compare_policies_statistically(
        config,
        policy_base="FIJA",
        policy_alt=pol_dyn,
        n_replicas=n_replicas_confirm,
        base_seed=seed0,
        assignment_base=opt_fixed["best_assignment"],
        assignment_alt=opt_dyn["best_assignment"],
    )
    comp["best_fixed_assignment_str"] = opt_fixed["best_assignment_str"]
    comp["best_dynamic_assignment_str"] = opt_dyn["best_assignment_str"]
    return comp


# ==============================================================================
# 6. EVALUACIÓN DE RESILIENCIA ANTE FALLA DE SCOOPTRAM (WHAT-IF)
# ==============================================================================
def evaluate_breakdown_resilience(
    config: SimulationConfig,
    breakdown: BreakdownConfig,
    policies: Sequence[str] = ("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
    n_replicas: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compara el desempeño Sin Falla vs. Con Falla bajo cada política usando CRN.
    Cuantifica pérdida de mineral, caída de ingreso neto, cola extra, ventilación pico,
    diésel/CO2 en colas y Resiliencia (%).
    """
    seed0 = int(config.global_cfg.semilla if base_seed is None else base_seed)
    bd_active = BreakdownConfig(
        event_id=breakdown.event_id,
        station_key=breakdown.station_key,
        mode=breakdown.mode,
        start_min=breakdown.start_min,
        duration_min=breakdown.duration_min,
        mtbf_hr=breakdown.mtbf_hr,
        mttr_min=breakdown.mttr_min,
        units_affected=breakdown.units_affected,
        active=True,
    )

    rows: List[Dict[str, object]] = []
    details: Dict[str, Dict[str, object]] = {}

    for pol in policies:
        pol_u = normalize_policy(pol)
        cfg_ok = config.clone()
        cfg_ok.global_cfg.avoid_down_zones = True
        cfg_ok.global_cfg.politica_despacho = pol_u
        cfg_ok.breakdowns = [b for b in cfg_ok.breakdowns if not b.active]

        cfg_fail = cfg_ok.clone()
        cfg_fail.breakdowns.append(bd_active)

        res_ok = run_replications(cfg_ok, n_replicas=n_replicas, base_seed=seed0)
        res_fail = run_replications(cfg_fail, n_replicas=n_replicas, base_seed=seed0)

        ore_ok, ore_fail = res_ok["ore_tn"]["mean"], res_fail["ore_tn"]["mean"]
        net_ok, net_fail = res_ok["net_revenue_usd"]["mean"], res_fail["net_revenue_usd"]["mean"]
        wait_ok, wait_fail = res_ok["truck_wait_hr"]["mean"], res_fail["truck_wait_hr"]["mean"]
        resilience_pct = (ore_fail / ore_ok * 100.0) if ore_ok > 0 else 0.0

        rows.append({
            "Código": pol_u,
            "Política": POLICY_LABELS.get(pol_u, pol_u),
            "Mineral Normal (t)": round(ore_ok, 2),
            "Mineral c/Falla (t)": round(ore_fail, 2),
            "Pérdida Mineral (t)": round(ore_fail - ore_ok, 2),
            "Resiliencia (%)": round(resilience_pct, 2),
            "Neto Normal (US$)": round(net_ok, 2),
            "Neto c/Falla (US$)": round(net_fail, 2),
            "Impacto Neto (US$/gdia)": round(net_fail - net_ok, 2),
            "Cola Extra (h/gdia)": round(wait_fail - wait_ok, 3),
            "Vent. Q Pico c/Falla (m3/min)": res_fail["abet"]["vent_q_peak_total_m3_min"],
            "Diésel Cola c/Falla (L/gdia)": res_fail["abet"]["diesel_wasted_wait_l_shift"],
            "CO2 Cola c/Falla (kg/gdia)": res_fail["abet"]["co2_wasted_wait_kg_shift"],
        })
        details[pol_u] = {"normal": res_ok, "breakdown": res_fail}

    return {
        "base_seed": seed0,
        "breakdown": bd_active,
        "summary_df": pd.DataFrame(rows),
        "details": details,
    }


# ==============================================================================
# 7. PRUEBA DE VERIFICACIÓN INTEGRAL (CLI)
# ==============================================================================
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1600)

    cfg_base = build_huancapeti_base_config(n_her=6, n_car=2, n_cot=7, policy="DIN_ETA_RITMO", seed=1)

    print("=" * 105)
    print(" 1. BENCHMARK SIMULTÁNEO DE LAS 5 POLÍTICAS (CRN, 30 RÉPLICAS, FLOTA 6-2-7)")
    print("=" * 105)
    bench = benchmark_all_policies(cfg_base, n_replicas=30, base_seed=1)
    print(bench["summary_df"].drop(columns=["Código"]).to_string(index=False))

    print("\n" + "=" * 105)
    print(" 2. CONTRASTE ESTADÍSTICO FORMAL (30 RÉPLICAS): FIJA vs. DIN_ETA_RITMO (6-2-7)")
    print("=" * 105)
    comp = compare_policies_statistically(
        cfg_base, policy_base="FIJA", policy_alt="DIN_ETA_RITMO", n_replicas=30, base_seed=1
    )
    print(comp["comparison_table"].to_string(index=False))
    print(
        f"Impacto Neto Anual: {comp['delta_net_usd_anio']:+,.2f} US$/año | "
        f"CO2 evitado: {comp['abet_comparison']['co2_avoided_t_year']:+.3f} t/año | "
        f"Vent. Q Pico: {comp['abet_comparison']['vent_q_peak_total_m3_min']:,.2f} m3/min "
        f"({comp['abet_comparison']['vent_q_peak_total_cfm']:,.1f} CFM)"
    )

    print("\n" + "=" * 105)
    print(" 3. PRUEBA DE RESILIENCIA ANTE FALLA DE SCOOPTRAM (60 min en COT, 15 RÉPLICAS)")
    print("=" * 105)
    bd_cot = BreakdownConfig(
        event_id="FALLA_COT_60M", station_key="COT", mode="SCHEDULED", start_min=180.0, duration_min=60.0, active=True
    )
    resil = evaluate_breakdown_resilience(cfg_base, breakdown=bd_cot, n_replicas=15, base_seed=1)
    print(resil["summary_df"].drop(columns=["Código"]).to_string(index=False))