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


# ==============================================================================
# 2. MÓDULO ABET: VENTILACIÓN (D.S. 023-2017-EM) + DIÉSEL / CO2 (IPCC/MINAM)
# ==============================================================================
def compute_abet_metrics(
    config: SimulationConfig,
    kpi_summary: Dict[str, object],
    baseline_wait_hr: Optional[float] = None,
) -> Dict[str, object]:
    """
    Calcula requerimientos normativos de ventilación (D.S. 023-2017-EM, Art. 252
    y Anexo 38) y consumo de diésel / emisiones de CO2 en ralentí (IPCC/MINAM).
      Q_equipos = factor_vent * HP * Dm * Fu  [m3/min]
      Diesel_ahorrado = delta_t_cola_h * C_ralenti
      CO2_evitado = Diesel_ahorrado * factor_co2
    """
    g = config.global_cfg
    active_zones = [z for z in config.zones if z.active]

    n_trucks = int(kpi_summary["n_trucks_total"])
    truck_fu = float(kpi_summary["truck_fu"])
    scoop_utils: Dict[str, float] = kpi_summary["scoop_utilization_by_zone"]

    # Ventilación de volquetes (m3/min)
    q_trucks = g.factor_vent_m3_min_hp * n_trucks * g.hp_volquete * g.dm_volquete * truck_fu

    # Ventilación de scooptrams por zona (m3/min)
    q_scoops_by_zone: Dict[str, float] = {}
    q_scoops_total = 0.0
    total_scoops = 0
    for z in active_zones:
        hp_sc = z.hp_scoop if z.hp_scoop is not None else g.hp_scoop
        fu_sc = float(scoop_utils.get(z.zone_id, 0.0))
        qz = g.factor_vent_m3_min_hp * z.n_scoops * hp_sc * g.dm_scoop * fu_sc
        q_scoops_by_zone[z.zone_id] = round(qz, 2)
        q_scoops_total += qz
        total_scoops += int(z.n_scoops)

    q_total = q_trucks + q_scoops_total
    q_total_cfm = q_total * 35.3146667

    cap_max = float(g.cap_max_ventilacion_m3_min)
    if cap_max > 0.0:
        vent_usage_pct = (q_total / cap_max) * 100.0
        vent_margin = cap_max - q_total
        vent_alert = q_total > cap_max
    else:
        vent_usage_pct = 0.0
        vent_margin = 0.0
        vent_alert = False

    # Diésel y CO2 consumidos en ralentí durante colas
    wait_hr = float(kpi_summary["total_truck_wait_hr"])
    diesel_queue_l_gdia = wait_hr * g.diesel_ralenti_l_hr
    diesel_queue_usd_gdia = diesel_queue_l_gdia * g.precio_diesel_usd_l
    co2_queue_kg_gdia = diesel_queue_l_gdia * g.factor_co2_kg_l

    # Ahorro respecto a escenario base (ej. Despacho Fijo)
    wait_saved_hr = (float(baseline_wait_hr) - wait_hr) if baseline_wait_hr is not None else 0.0
    diesel_saved_l_gdia = wait_saved_hr * g.diesel_ralenti_l_hr
    diesel_saved_usd_gdia = diesel_saved_l_gdia * g.precio_diesel_usd_l
    co2_avoided_kg_gdia = diesel_saved_l_gdia * g.factor_co2_kg_l

    guardias = int(g.guardias_anio)
    return {
        "n_trucks": n_trucks,
        "n_scoops_total": total_scoops,
        "truck_fu": round(truck_fu, 4),
        "q_trucks_m3_min": round(q_trucks, 2),
        "q_scoops_m3_min": round(q_scoops_total, 2),
        "q_scoops_by_zone_m3_min": q_scoops_by_zone,
        "q_total_m3_min": round(q_total, 2),
        "q_total_cfm": round(q_total_cfm, 1),
        "cap_max_vent_m3_min": round(cap_max, 2),
        "vent_usage_pct": round(vent_usage_pct, 2),
        "vent_margin_m3_min": round(vent_margin, 2),
        "vent_alert": vent_alert,
        "wait_hr_gdia": round(wait_hr, 4),
        "diesel_queue_l_gdia": round(diesel_queue_l_gdia, 2),
        "diesel_queue_usd_gdia": round(diesel_queue_usd_gdia, 2),
        "co2_queue_kg_gdia": round(co2_queue_kg_gdia, 2),
        "co2_queue_t_anio": round((co2_queue_kg_gdia * guardias) / 1000.0, 2),
        "wait_saved_hr_gdia": round(wait_saved_hr, 4),
        "diesel_saved_l_gdia": round(diesel_saved_l_gdia, 2),
        "diesel_saved_l_anio": round(diesel_saved_l_gdia * guardias, 2),
        "diesel_saved_usd_gdia": round(diesel_saved_usd_gdia, 2),
        "diesel_saved_usd_anio": round(diesel_saved_usd_gdia * guardias, 2),
        "co2_avoided_kg_gdia": round(co2_avoided_kg_gdia, 2),
        "co2_avoided_t_anio": round((co2_avoided_kg_gdia * guardias) / 1000.0, 2),
    }


# ==============================================================================
# 3. EJECUTOR DE RÉPLICAS MONTECARLO (CON IC 95% Y CUOTAS POR ZONA)
# ==============================================================================
def run_replications(
    config: SimulationConfig,
    n_replicas: Optional[int] = None,
    base_seed: int = 1,
    seed_step: int = 1,
    confidence: Optional[float] = None,
) -> Dict[str, object]:
    """Ejecuta n_replicas corridas independientes y calcula IC t-Student y ABET."""
    g = config.global_cfg
    pol = normalize_policy(g.politica_despacho)
    n_rep = max(1, int(g.num_replicas if n_replicas is None else n_replicas))
    conf = float(g.nivel_confianza if confidence is None else confidence)
    active_zones = [z for z in config.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]

    replica_rows: List[Dict[str, object]] = []
    raw_kpis: List[Dict[str, object]] = []

    for i in range(n_rep):
        sd = base_seed + i * seed_step
        kpi = SubHaulDESEngine(config, seed_override=sd).run()
        raw_kpis.append(kpi)

        eff_horizon_sec = float(kpi["effective_horizon_sec"])
        row: Dict[str, object] = {
            "replica": i + 1,
            "seed": sd,
            "effective_horizon_sec": eff_horizon_sec,
            "effective_horizon_hr": eff_horizon_sec / 3600.0,
            "ore_total_tn": float(kpi["ore_total_tn"]),
            "waste_total_tn": float(kpi["waste_total_tn"]),
            "gross_revenue_usd": float(kpi["gross_revenue_usd"]),
            "scoop_op_cost_usd": float(kpi["scoop_op_cost_usd"]),
            "scoop_idle_cost_usd": float(kpi["scoop_idle_cost_usd"]),
            "truck_op_cost_usd": float(kpi["truck_op_cost_usd"]),
            "truck_wait_cost_usd": float(kpi["truck_wait_cost_usd"]),
            "fleet_op_cost_usd": float(kpi["fleet_op_cost_usd"]),
            "waste_extra_cost_usd": float(kpi["waste_extra_cost_usd"]),
            "marginal_fleet_cost_usd": float(kpi["marginal_fleet_cost_usd"]),
            "total_opex_usd": float(kpi["total_opex_usd"]),
            "net_revenue_usd": float(kpi["net_revenue_usd"]),
            "unit_cost_usd_tn": float(kpi["unit_cost_usd_tn"]),
            "total_truck_wait_hr": float(kpi["total_truck_wait_hr"]),
            "scale_wait_hr": float(kpi["scale_wait_hr"]),
            "scale_utilization": float(kpi["scale_utilization"]),
            "truck_fu": float(kpi["truck_fu"]),
            "target_met": bool(kpi["target_met"]),
        }
        for zid in zone_ids:
            row[f"ore_{zid}"] = kpi["ore_by_zone"][zid]
            row[f"util_{zid}"] = kpi["scoop_utilization_by_zone"][zid]
            row[f"wait_hr_{zid}"] = kpi["wait_hr_by_zone"][zid]
            row[f"max_q_{zid}"] = kpi["max_queue_by_station"][zid]
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

    ore_zone_means = {zid: round(float(df_rep[f"ore_{zid}"].mean()), 2) for zid in zone_ids}
    util_zone_means = {zid: round(float(df_rep[f"util_{zid}"].mean()), 4) for zid in zone_ids}
    max_q_zone_means = {zid: round(float(df_rep[f"max_q_{zid}"].mean()), 2) for zid in zone_ids}

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
        config,
        {
            "n_trucks_total": int(raw_kpis[0]["n_trucks_total"]),
            "truck_fu": fu_ci["mean"],
            "scoop_utilization_by_zone": util_zone_means,
            "total_truck_wait_hr": wait_ci["mean"],
        },
    )

    return {
        "n_replicas": n_rep,
        "confidence": conf,
        "policy": pol,
        "policy_label": POLICY_LABELS.get(pol, pol),
        "stop_mode": g.modo_parada,
        "effective_horizon_hr": horizon_hr_ci,
        "n_trucks_total": int(raw_kpis[0]["n_trucks_total"]),
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
        "prob_target_met_pct": round(float(df_rep["target_met"].mean() * 100.0), 2),
        "target_met_mean": ore_ci["mean"] >= g.meta_mineral_gdia,
        "ore_by_zone_mean": ore_zone_means,
        "scoop_util_by_zone_mean": util_zone_means,
        "max_queue_by_zone_mean": max_q_zone_means,
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
    base_seed: int = 1,
    assignment_base: Optional[Dict[str, int]] = None,
    assignment_alt: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    """Compara dos configuraciones usando Common Random Numbers (CRN)."""
    pol_a = normalize_policy(policy_base)
    pol_b = normalize_policy(policy_alt)

    cfg_a = config.clone()
    cfg_a.global_cfg.politica_despacho = pol_a
    if assignment_base is not None:
        cfg_a = cfg_a.set_fleet_assignment(assignment_base)

    cfg_b = config.clone()
    cfg_b.global_cfg.politica_despacho = pol_b
    if assignment_alt is not None:
        cfg_b = cfg_b.set_fleet_assignment(assignment_alt)

    res_a = run_replications(cfg_a, n_replicas=n_replicas, base_seed=base_seed)
    res_b = run_replications(cfg_b, n_replicas=n_replicas, base_seed=base_seed)

    df_a, df_b = res_a["replicas_df"], res_b["replicas_df"]
    conf = config.global_cfg.nivel_confianza
    guardias = int(config.global_cfg.guardias_anio)

    test_ore = _safe_hypothesis_tests(df_a["ore_total_tn"].values, df_b["ore_total_tn"].values, conf)
    test_net = _safe_hypothesis_tests(df_a["net_revenue_usd"].values, df_b["net_revenue_usd"].values, conf)
    test_wait = _safe_hypothesis_tests(df_a["total_truck_wait_hr"].values, df_b["total_truck_wait_hr"].values, conf)

    abet_b_vs_a = compute_abet_metrics(
        cfg_b,
        {
            "n_trucks_total": res_b["n_trucks_total"],
            "truck_fu": res_b["truck_fu"]["mean"],
            "scoop_utilization_by_zone": res_b["scoop_util_by_zone_mean"],
            "total_truck_wait_hr": res_b["truck_wait_hr"]["mean"],
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
    ]

    return {
        "policy_base": pol_a,
        "policy_base_label": POLICY_LABELS.get(pol_a, pol_a),
        "policy_alt": pol_b,
        "policy_alt_label": POLICY_LABELS.get(pol_b, pol_b),
        "result_base": res_a,
        "result_alt": res_b,
        "test_ore": test_ore,
        "test_net": test_net,
        "test_wait": test_wait,
        "delta_net_usd_gdia": round(delta_net_gdia, 2),
        "delta_net_usd_anio": delta_net_anio,
        "abet_comparison": abet_b_vs_a,
        "comparison_table": pd.DataFrame(summary_rows),
    }


def benchmark_all_policies(
    config: SimulationConfig,
    policies: Sequence[str] = ("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
    n_replicas: Optional[int] = None,
    base_seed: int = 1,
) -> Dict[str, object]:
    """
    Ejecuta en simultáneo todas las políticas indicadas usando Common Random Numbers (CRN)
    sobre la flota configurada y construye una matriz comparativa ejecutiva.
    """
    active_zones = [z for z in config.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]

    results_by_policy: Dict[str, Dict[str, object]] = {}
    for pol in policies:
        pol_n = normalize_policy(pol)
        cfg_p = config.clone()
        cfg_p.global_cfg.politica_despacho = pol_n
        results_by_policy[pol_n] = run_replications(cfg_p, n_replicas=n_replicas, base_seed=base_seed)

    # Referencia base para calcular deltas y ahorro de CO2 (FIJA si está presente, si no la primera)
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
        cfg_p.global_cfg.politica_despacho = pol_n
        abet_vs_base = compute_abet_metrics(
            cfg_p,
            {
                "n_trucks_total": res["n_trucks_total"],
                "truck_fu": res["truck_fu"]["mean"],
                "scoop_utilization_by_zone": res["scoop_util_by_zone_mean"],
                "total_truck_wait_hr": wait_mean,
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
            "CO2 Evitado vs Fija (t/año)": abet_vs_base["co2_avoided_t_anio"],
        }
        for zid in zone_ids:
            row[f"Ore_{zid} (t)"] = res["ore_by_zone_mean"][zid]
        rows.append(row)

    return {
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
    n_replicas_screen: int = 5,
    n_replicas_confirm: int = 20,
    top_k_confirm: int = 3,
    min_trucks_per_zone: int = 1,
    require_target_met: bool = True,
    min_zone_quota_pct: float = 0.0,
    max_exhaustive_candidates: int = 120,
    base_seed: int = 1,
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
    g.usar_tope_planta = bool(use_plant_cap)
    g.cap_planta_tn = float(plant_cap_tn) if use_plant_cap else 0.0
    if policy is not None:
        g.politica_despacho = normalize_policy(policy)
    else:
        g.politica_despacho = normalize_policy(g.politica_despacho)
    obj = (g.funcion_objetivo if objective is None else objective).strip().upper()

    f_min = int(g.flota_min if fleet_min is None else fleet_min)
    f_max = int(g.flota_max if fleet_max is None else fleet_max)
    if f_max < f_min:
        f_min, f_max = f_max, f_min

    active_zones = [z for z in cfg_work.zones if z.active]
    zone_ids = [z.zone_id for z in active_zones]
    n_zones = len(zone_ids)
    fleet_sizes = list(range(f_min, f_max + 1))

    all_confirmed_records: List[Dict[str, object]] = []
    best_per_fleet_records: List[Dict[str, object]] = []
    best_per_fleet_full: Dict[int, Dict[str, object]] = {}

    for step_idx, n_fleet in enumerate(fleet_sizes):
        if progress_callback:
            progress_callback(step_idx / max(1, len(fleet_sizes)), f"Optimizando flota N = {n_fleet} volquetes...")

        eff_min = min_trucks_per_zone if n_fleet >= n_zones * min_trucks_per_zone else 0
        rem = n_fleet - n_zones * eff_min
        n_comb = math.comb(rem + n_zones - 1, n_zones - 1) if n_zones > 1 else 1
        screen_cache: Dict[Tuple[int, ...], Dict[str, object]] = {}

        def eval_screen(tup: Tuple[int, ...]) -> Dict[str, object]:
            if tup not in screen_cache:
                asig_dict = {zid: int(cnt) for zid, cnt in zip(zone_ids, tup)}
                screen_cache[tup] = run_replications(
                    cfg_work.set_fleet_assignment(asig_dict),
                    n_replicas=n_replicas_screen,
                    base_seed=base_seed,
                )
            return screen_cache[tup]

        if n_comb <= max_exhaustive_candidates:
            for tup in _generate_compositions(n_fleet, n_zones, min_per_zone=eff_min):
                eval_screen(tup)
        else:
            current = _initial_proportional_assignment(cfg_work, n_fleet, min_per_zone=eff_min)
            cur_score = _objective_score(
                eval_screen(current), obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct
            )
            improved, it = True, 0
            while improved and it < 25:
                improved, it = False, it + 1
                best_nb, best_nb_score = current, cur_score
                for i in range(n_zones):
                    if current[i] <= eff_min:
                        continue
                    for j in range(n_zones):
                        if i == j:
                            continue
                        nb = list(current)
                        nb[i] -= 1
                        nb[j] += 1
                        nb_tup = tuple(nb)
                        nb_score = _objective_score(
                            eval_screen(nb_tup), obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct
                        )
                        if nb_score > best_nb_score:
                            best_nb, best_nb_score = nb_tup, nb_score
                if best_nb != current:
                    current, cur_score, improved = best_nb, best_nb_score, True

        ranked_screen = sorted(
            screen_cache.items(),
            key=lambda kv: _objective_score(kv[1], obj, g.meta_mineral_gdia, require_target_met, min_zone_quota_pct),
            reverse=True,
        )
        top_tuples = [tup for tup, _ in ranked_screen[: max(1, min(top_k_confirm, len(ranked_screen)))]]

        confirmed_for_fleet: List[Tuple[Tuple[int, ...], Dict[str, object]]] = []
        for tup in top_tuples:
            if n_replicas_confirm <= n_replicas_screen:
                res_conf = screen_cache[tup]
            else:
                asig_dict = {zid: int(cnt) for zid, cnt in zip(zone_ids, tup)}
                res_conf = run_replications(
                    cfg_work.set_fleet_assignment(asig_dict),
                    n_replicas=n_replicas_confirm,
                    base_seed=base_seed,
                )
            confirmed_for_fleet.append((tup, res_conf))

            rec: Dict[str, object] = {
                "Flota (N)": n_fleet,
                "Asignación": res_conf["assignment_str"],
                "Política": POLICY_LABELS.get(g.politica_despacho, g.politica_despacho),
                "Mineral (t/gdia)": round(res_conf["ore_tn"]["mean"], 2),
                "IC95 Mineral (±)": round(res_conf["ore_tn"]["ci_half"], 2),
                "Cumple Meta (%)": res_conf["prob_target_met_pct"],
                "Cumpl. Mín. Cuota Zona (%)": res_conf["min_quota_compliance_pct"],
                "Ingreso Neto (US$/gdia)": round(res_conf["net_revenue_usd"]["mean"], 2),
                "IC95 Neto (±)": round(res_conf["net_revenue_usd"]["ci_half"], 2),
                "OPEX Total (US$/gdia)": round(res_conf["total_opex_usd"]["mean"], 2),
                "Costo Flota Volq. (US$/gdia)": round(res_conf["fleet_op_cost_usd"]["mean"], 2),
                "Costo Unitario (US$/t)": round(res_conf["unit_cost_usd_tn"]["mean"], 3),
                "Cola Total (h/gdia)": round(res_conf["truck_wait_hr"]["mean"], 3),
                "Fu Volquetes": round(res_conf["truck_fu"]["mean"], 4),
                "Ventilación Q (m3/min)": res_conf["abet"]["q_total_m3_min"],
                "Alerta Vent.": "SÍ" if res_conf["abet"]["vent_alert"] else "NO",
                "CO2 Cola (kg/gdia)": res_conf["abet"]["co2_queue_kg_gdia"],
            }
            for zid in zone_ids:
                rec[f"Ore_{zid} (t)"] = res_conf["ore_by_zone_mean"][zid]
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
    df_curve["Δ Ventilación/Volq (m3/min)"] = df_curve["Ventilación Q (m3/min)"].diff().round(2).fillna(0.0)

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
        "objective": obj,
        "policy": g.politica_despacho,
        "policy_label": POLICY_LABELS.get(g.politica_despacho, g.politica_despacho),
        "use_plant_cap": bool(use_plant_cap),
        "plant_cap_tn": float(plant_cap_tn),
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
    n_replicas_screen: int = 5,
    n_replicas_confirm: int = 30,
    min_zone_quota_pct: float = 90.0,
    base_seed: int = 1,
) -> Dict[str, object]:
    """
    Contraste metodológicamente justo:
    Optimiza por separado el Mejor Escenario Fijo y el Mejor Escenario Dinámico
    para la misma flota N y restricciones, y luego corre las pruebas formales
    (t-Student y Mann-Whitney U) entre ambos ganadores.
    """
    pol_dyn = normalize_policy(policy_dynamic)
    opt_fixed = optimize_fleet_and_assignment(
        config,
        fleet_min=fleet_size,
        fleet_max=fleet_size,
        policy="FIJA",
        objective=objective,
        n_replicas_screen=n_replicas_screen,
        n_replicas_confirm=max(10, n_replicas_confirm // 2),
        top_k_confirm=3,
        require_target_met=True,
        min_zone_quota_pct=min_zone_quota_pct,
        base_seed=base_seed,
    )
    opt_dyn = optimize_fleet_and_assignment(
        config,
        fleet_min=fleet_size,
        fleet_max=fleet_size,
        policy=pol_dyn,
        objective=objective,
        n_replicas_screen=n_replicas_screen,
        n_replicas_confirm=max(10, n_replicas_confirm // 2),
        top_k_confirm=3,
        require_target_met=True,
        min_zone_quota_pct=min_zone_quota_pct,
        base_seed=base_seed,
    )

    comp = compare_policies_statistically(
        config,
        policy_base="FIJA",
        policy_alt=pol_dyn,
        n_replicas=n_replicas_confirm,
        base_seed=base_seed,
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
    base_seed: int = 1,
) -> Dict[str, object]:
    """
    Compara el desempeño Sin Falla vs. Con Falla bajo cada política usando CRN.
    Cuantifica pérdida de mineral, caída de ingreso neto, cola extra y Resiliencia (%).
    """
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
        cfg_ok.global_cfg.politica_despacho = pol_u
        cfg_ok.breakdowns = [b for b in cfg_ok.breakdowns if not b.active]

        cfg_fail = cfg_ok.clone()
        cfg_fail.breakdowns.append(bd_active)

        res_ok = run_replications(cfg_ok, n_replicas=n_replicas, base_seed=base_seed)
        res_fail = run_replications(cfg_fail, n_replicas=n_replicas, base_seed=base_seed)

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
        })
        details[pol_u] = {"normal": res_ok, "breakdown": res_fail}

    return {
        "breakdown": bd_active,
        "summary_df": pd.DataFrame(rows),
        "details": details,
    }


# ==============================================================================
# 7. PRUEBA DE VERIFICACIÓN INTEGRAL (CLI)
# ==============================================================================
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1300)

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
    print(f"Impacto Neto Anual: {comp['delta_net_usd_anio']:+,.2f} US$/año | CO2 evitado: {comp['abet_comparison']['co2_avoided_t_anio']:+.2f} t/año")

    print("\n" + "=" * 105)
    print(" 3. PRUEBA DE RESILIENCIA ANTE FALLA DE SCOOPTRAM (60 min en COT, 15 RÉPLICAS)")
    print("=" * 105)
    bd_cot = BreakdownConfig(
        event_id="FALLA_COT_60M", station_key="COT", mode="SCHEDULED", start_min=180.0, duration_min=60.0, active=True
    )
    resil = evaluate_breakdown_resilience(cfg_base, breakdown=bd_cot, n_replicas=15, base_seed=1)
    print(resil["summary_df"].drop(columns=["Código"]).to_string(index=False))