"""
SubHaul-Opt — Interfaz Web Interactiva en Streamlit.
Feria ABET UNI 2026-II | Métodos de Explotación Subterránea (MI753-A)

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from motor_des import (
    M3_MIN_TO_CFM,
    POLICY_LABELS,
    POLICY_OPTIONS,
    BreakdownConfig,
    DistributionSpec,
    SimulationConfig,
    SubHaulDESEngine,
    ZoneConfig,
    build_huancapeti_base_config,
    export_excel_template,
    load_config_from_excel,
    normalize_policy,
)
from optimizador import (
    benchmark_all_policies,
    compare_optimal_fixed_vs_dynamic,
    compare_policies_statistically,
    compute_abet_metrics,
    evaluate_breakdown_resilience,
    optimize_fleet_and_assignment,
    run_replications,
)
from visualizador_2d import build_2d_comparison_html, build_2d_viewer_html

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA, ESQUEMA CANÓNICO DE 9 RUTAS Y ESTADO DE SESIÓN
# ==============================================================================
st.set_page_config(
    page_title="SubHaul-Opt | Sim-Opt Minería Subterránea",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DYNAMIC_POLICY_OPTIONS = [p for p in POLICY_OPTIONS if p != "FIJA"]
STOP_OPTIONS = ["HARD", "SOFT"]
OBJ_OPTIONS = ["INGRESO_NETO", "TONELAJE", "COSTO_UNITARIO"]

# Esquema canónico de 9 columnas de ruta por zona (ancladas en bocamina/acceso)
ROUTE_COLUMNS: List[str] = [
    "dist_inicial",
    "dist_bajada_vacio",
    "dist_carguio",
    "dist_subida_cargado",
    "dist_hacia_balanza",
    "dist_balanza_a_cancha",
    "dist_hacia_desmontera",
    "dist_retorno_desde_mineral",
    "dist_retorno_desde_desmonte",
]


def _policy_fmt(code: str) -> str:
    return POLICY_LABELS.get(code, code)


def _fmt_hm(hours: float) -> str:
    """Formatea horas decimales a texto legible de ingeniería (ej. '8 h 14 min (8.23 h)')."""
    total_min = int(round(max(0.0, float(hours)) * 60.0))
    hh, mm = divmod(total_min, 60)
    return f"{hh} h {mm:02d} min ({hours:.2f} h)"


def _sync_meta_from_zones(cfg: SimulationConfig) -> float:
    """
    Deriva automáticamente `meta_mineral_gdia` como la suma de `quota_mineral_tn`
    de las zonas activas. Si todas las cuotas son 0, conserva el valor de respaldo.
    """
    quota_sum = sum(float(z.quota_mineral_tn) for z in cfg.zones if z.active)
    if quota_sum > 0.0:
        cfg.global_cfg.meta_mineral_gdia = quota_sum
    return float(cfg.global_cfg.meta_mineral_gdia)


def _render_info_kpi(container: object, label: str, value: str, subtitle: str) -> None:
    """
    Renderiza una tarjeta KPI limpia con anotación inferior gris informativa
    SIN ninguna flecha (ni ↑ ni ↓).
    """
    container.markdown(
        f"""
        <div style="padding: 0.25rem 0 0.5rem 0;">
            <div style="font-size: 0.875rem; color: var(--text-color); opacity: 0.85; margin-bottom: 0.2rem;">
                {label}
            </div>
            <div style="font-size: 2.1rem; font-weight: 600; line-height: 1.2; color: var(--text-color);">
                {value}
            </div>
            <div style="display: inline-block; margin-top: 0.35rem; padding: 0.15rem 0.55rem;
                        border-radius: 999px; font-size: 0.82rem; font-weight: 500;
                        background-color: rgba(128, 128, 128, 0.14); color: var(--text-color); opacity: 0.85;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _config_matches_current_schema(cfg: object) -> bool:
    """Verifica que una configuración en session_state posea el esquema vigente de 9 rutas y ventilación por zona."""
    zones = getattr(cfg, "zones", None)
    if not zones:
        return False
    return all(
        all(hasattr(z, col) for col in ROUTE_COLUMNS) and hasattr(z, "cap_max_ventilacion_m3_min")
        for z in zones
    )


def _init_state() -> None:
    if "config" not in st.session_state or not _config_matches_current_schema(st.session_state.config):
        cfg_init = build_huancapeti_base_config(
            n_her=6, n_car=2, n_cot=7, policy="DIN_ETA_RITMO", prob_mode="FORECAST", seed=25
        )
        _sync_meta_from_zones(cfg_init)
        st.session_state.config = cfg_init
        st.session_state.dash_res = None
        st.session_state.opt_res = None
        st.session_state.comp_res = None
        st.session_state.bench_res = None
        st.session_state.resil_res = None
        st.session_state.sim2d_res = None
    if "cfg_version" not in st.session_state:
        st.session_state.cfg_version = 0
    if "last_seed" not in st.session_state:
        st.session_state.last_seed = int(st.session_state.config.global_cfg.semilla)
    if "last_meta_gdia" not in st.session_state:
        st.session_state.last_meta_gdia = float(_sync_meta_from_zones(st.session_state.config))
    for key in ("dash_res", "opt_res", "comp_res", "bench_res", "resil_res", "sim2d_res"):
        if key not in st.session_state:
            st.session_state[key] = None


def _load_new_config(cfg: SimulationConfig) -> None:
    """Reemplaza la configuración activa, sincroniza la meta diaria y fuerza refresco de widgets."""
    cfg.global_cfg.politica_despacho = normalize_policy(cfg.global_cfg.politica_despacho)
    cfg.global_cfg.avoid_down_zones = True
    _sync_meta_from_zones(cfg)
    st.session_state.config = cfg
    st.session_state.cfg_version += 1
    st.session_state.last_seed = int(cfg.global_cfg.semilla)
    st.session_state.last_meta_gdia = float(cfg.global_cfg.meta_mineral_gdia)
    st.session_state.dash_res = None
    st.session_state.opt_res = None
    st.session_state.comp_res = None
    st.session_state.bench_res = None
    st.session_state.resil_res = None
    st.session_state.sim2d_res = None


def _template_bytes(cfg: SimulationConfig) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "plantilla_subhaul_opt.xlsx"
        export_excel_template(path, cfg)
        return path.read_bytes()


def _format_route_chain(chain: List[str]) -> str:
    cleaned = [str(seg).strip() for seg in chain if str(seg).strip()]
    return "|".join(cleaned) if cleaned else "0"


def _parse_chain_str(val: object, fallback: str = "0") -> List[str]:
    """
    Convierte una celda del editor en lista de claves/constantes.
    Acepta claves separadas por '|' y constantes numéricas en segundos (incluido 0).
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return [fallback] if fallback != "" else []
    if pd.isna(val):
        return [fallback] if fallback != "" else []
    text = str(val).strip()
    if not text:
        return [fallback] if fallback != "" else []
    parts = [p.strip() for p in text.split("|") if p.strip()]
    return parts if parts else ([fallback] if fallback != "" else [])


def _zones_to_df(zones: List[ZoneConfig]) -> pd.DataFrame:
    rows = []
    for z in zones:
        rows.append({
            "zone_id": z.zone_id,
            "zone_name": z.zone_name,
            "active": bool(z.active),
            "n_trucks": int(z.n_trucks),
            "n_scoops": int(z.n_scoops),
            "prob_mineral": float(z.prob_mineral),
            "quota_mineral_tn": float(z.quota_mineral_tn),
            "max_quota_mineral_tn": float(z.max_quota_mineral_tn),
            "ore_price_usd_tn": float(z.ore_price_usd_tn),
            "waste_tonnage_dist": str(z.waste_tonnage_dist),
            "cap_max_ventilacion_m3_min": float(z.cap_max_ventilacion_m3_min),
            # 9 columnas de ruta en orden lógico del ciclo (ancladas en bocamina)
            "dist_inicial": _format_route_chain(z.dist_inicial),
            "dist_bajada_vacio": _format_route_chain(z.dist_bajada_vacio),
            "dist_carguio": str(z.dist_carguio).strip() if str(z.dist_carguio).strip() else "0",
            "dist_subida_cargado": _format_route_chain(z.dist_subida_cargado),
            "dist_hacia_balanza": _format_route_chain(z.dist_hacia_balanza),
            "dist_balanza_a_cancha": _format_route_chain(z.dist_balanza_a_cancha),
            "dist_hacia_desmontera": _format_route_chain(z.dist_hacia_desmontera),
            "dist_retorno_desde_mineral": _format_route_chain(z.dist_retorno_desde_mineral),
            "dist_retorno_desde_desmonte": _format_route_chain(z.dist_retorno_desde_desmonte),
            "entry_start_sec": float(z.entry_start_sec),
            "entry_interval_sec": float(z.entry_interval_sec),
        })
    return pd.DataFrame(rows)


def _df_to_zones(df: pd.DataFrame, prev_zones: List[ZoneConfig]) -> List[ZoneConfig]:
    prev_map = {z.zone_id: z for z in prev_zones}
    zones: List[ZoneConfig] = []
    for idx, r in df.iterrows():
        raw_id = "" if pd.isna(r.get("zone_id")) else str(r.get("zone_id")).strip().upper()
        if not raw_id:
            raw_id = f"ZONA_{idx + 1}"
        old = prev_map.get(raw_id)

        waste_raw = r.get("waste_tonnage_dist", "20.0")
        waste_str = "20.0" if pd.isna(waste_raw) or str(waste_raw).strip() == "" else str(waste_raw).strip()
        try:
            waste_val: object = float(waste_str)
        except ValueError:
            waste_val = waste_str

        carguio_raw = r.get("dist_carguio", "0")
        carguio_str = "0" if pd.isna(carguio_raw) or str(carguio_raw).strip() == "" else str(carguio_raw).strip()

        zones.append(
            ZoneConfig(
                zone_id=raw_id,
                zone_name=raw_id if pd.isna(r.get("zone_name")) or not str(r.get("zone_name")).strip() else str(r.get("zone_name")).strip(),
                active=True if pd.isna(r.get("active")) else bool(r.get("active")),
                n_trucks=1 if pd.isna(r.get("n_trucks")) else max(0, int(r.get("n_trucks"))),
                n_scoops=1 if pd.isna(r.get("n_scoops")) else max(1, int(r.get("n_scoops"))),
                prob_mineral=0.75 if pd.isna(r.get("prob_mineral")) else float(np.clip(r.get("prob_mineral"), 0.0, 1.0)),
                quota_mineral_tn=0.0 if pd.isna(r.get("quota_mineral_tn")) else max(0.0, float(r.get("quota_mineral_tn"))),
                max_quota_mineral_tn=0.0 if pd.isna(r.get("max_quota_mineral_tn")) else max(0.0, float(r.get("max_quota_mineral_tn"))),
                ore_price_usd_tn=60.0 if pd.isna(r.get("ore_price_usd_tn")) else float(r.get("ore_price_usd_tn")),
                waste_tonnage_dist=waste_val,
                ore_tonnage_dist=old.ore_tonnage_dist if old else None,
                entry_start_sec=900.0 if pd.isna(r.get("entry_start_sec")) else max(0.0, float(r.get("entry_start_sec"))),
                entry_interval_sec=240.0 if pd.isna(r.get("entry_interval_sec")) else max(0.0, float(r.get("entry_interval_sec"))),
                dist_inicial=_parse_chain_str(r.get("dist_inicial"), "0"),
                dist_bajada_vacio=_parse_chain_str(r.get("dist_bajada_vacio"), "0"),
                dist_carguio=carguio_str,
                dist_subida_cargado=_parse_chain_str(r.get("dist_subida_cargado"), "0"),
                dist_hacia_balanza=_parse_chain_str(r.get("dist_hacia_balanza"), "0"),
                dist_balanza_a_cancha=_parse_chain_str(r.get("dist_balanza_a_cancha"), "0"),
                dist_hacia_desmontera=_parse_chain_str(r.get("dist_hacia_desmontera"), "0"),
                dist_retorno_desde_mineral=_parse_chain_str(r.get("dist_retorno_desde_mineral"), "0"),
                dist_retorno_desde_desmonte=_parse_chain_str(r.get("dist_retorno_desde_desmonte"), "0"),
                facility_name=old.facility_name if old else f"SCOOP_{raw_id}",
                queue_name=old.queue_name if old else f"Q_{raw_id}",
                hp_scoop=old.hp_scoop if old else None,
                cap_max_ventilacion_m3_min=0.0 if pd.isna(r.get("cap_max_ventilacion_m3_min")) else max(0.0, float(r.get("cap_max_ventilacion_m3_min"))),
            )
        )
    return zones


def _validate_route_token(token: str, distributions: Dict[str, DistributionSpec]) -> bool:
    """Valida que un token de ruta sea una clave registrada o un número >= 0 (incluido 0)."""
    tok = token.strip()
    if not tok:
        return True
    if tok in distributions:
        return True
    try:
        val = float(tok)
        return (not math.isnan(val)) and (not math.isinf(val)) and val >= 0.0
    except ValueError:
        return False


def _validate_zones_routes(
    zones: List[ZoneConfig],
    distributions: Dict[str, DistributionSpec],
) -> List[str]:
    """Valida las 9 columnas de ruta y waste_tonnage_dist de todas las zonas activas."""
    errors: List[str] = []
    for z in zones:
        if not z.active:
            continue
        for col in ROUTE_COLUMNS:
            raw_val = getattr(z, col, "")
            parts = raw_val if isinstance(raw_val, list) else str(raw_val).split("|")
            for raw_part in parts:
                for tok in str(raw_part).split("|"):
                    tok = tok.strip()
                    if tok and not _validate_route_token(tok, distributions):
                        errors.append(
                            f"Zona **{z.zone_id}** → `{col}`: referencia inválida `{tok}`. "
                            "Usa una clave de distribución registrada o un número ≥ 0 en segundos (ej. `0`, `180`, `T4|T4HKC|T9`)."
                        )
        waste_tok = str(z.waste_tonnage_dist).strip()
        if waste_tok and not _validate_route_token(waste_tok, distributions):
            errors.append(
                f"Zona **{z.zone_id}** → `waste_tonnage_dist`: referencia inválida `{waste_tok}`. "
                "Ingresa un tonelaje numérico ≥ 0 o una clave de distribución registrada."
            )
    return errors


def _validate_uploaded_excel(filepath: Path) -> Tuple[bool, str]:
    """
    Valida la estructura de una plantilla Excel antes de cargarla.
    Detecta plantillas antiguas o con columnas de ruta incompatibles y devuelve
    un mensaje claro en español para la interfaz en lugar de lanzar KeyError.
    """
    try:
        xls = pd.ExcelFile(filepath, engine="openpyxl")
    except Exception as exc:
        return False, f"No se pudo abrir el archivo Excel (.xlsx): {exc}"

    required_sheets = {"1_Parametros_Globales", "2_Zonas"}
    missing_sheets = sorted(required_sheets - set(xls.sheet_names))
    if missing_sheets:
        return (
            False,
            f"La plantilla Excel no contiene las hojas obligatorias: {', '.join(missing_sheets)}. "
            "Descarga la plantilla base actualizada con el botón 'Descargar Plantilla Excel Actual'.",
        )

    df_z = pd.read_excel(xls, sheet_name="2_Zonas")
    cols = [str(c).strip() for c in df_z.columns]

    if "zone_id" not in cols or df_z.empty:
        return (
            False,
            "La hoja `2_Zonas` debe contener la columna `zone_id` y al menos una zona configurada.",
        )

    missing_routes = [c for c in ROUTE_COLUMNS if c not in cols]
    obsolete_routes = [
        c for c in cols
        if (c.startswith("dist_") and c not in ROUTE_COLUMNS) or c.startswith("skip_return")
    ]

    if missing_routes or obsolete_routes:
        parts = [
            "La plantilla subida pertenece a una versión anterior o tiene columnas de ruta incompatibles "
            "con el esquema vigente de 9 rutas ancladas en bocamina."
        ]
        if obsolete_routes:
            parts.append(f"Columnas antiguas/no reconocidas detectadas: `{', '.join(obsolete_routes)}`.")
        if missing_routes:
            parts.append(f"Columnas obligatorias faltantes: `{', '.join(missing_routes)}`.")
        parts.append(
            "Solución: descarga la plantilla vigente mediante **📥 Descargar Plantilla Excel Actual** "
            "y traslada tus rutas a las 9 columnas del esquema actual."
        )
        return False, " ".join(parts)

    return True, ""


def _build_ventilation_zone_df(cfg: SimulationConfig, abet: Dict[str, object]) -> pd.DataFrame:
    """Construye la tabla de caudal incremental requerido por zona (flota modelada)."""
    g = cfg.global_cfg
    rows = []
    f_vent = float(g.factor_vent_m3_min_hp)
    hp_volq = float(g.hp_volquete)

    for z in [z for z in cfg.zones if z.active]:
        zid = z.zone_id
        hp_sc = float(z.hp_scoop) if z.hp_scoop is not None else float(g.hp_scoop)
        peak_v = int(abet["underground_trucks_peak_by_zone"].get(zid, 0))
        avg_v = float(abet["underground_trucks_avg_by_zone"].get(zid, 0.0))
        q_peak = float(abet["vent_q_peak_m3_min_by_zone"].get(zid, 0.0))
        q_peak_cfm = float(abet["vent_q_peak_cfm_by_zone"].get(zid, 0.0))
        # Caudal promedio interior mina únicamente como dato informativo
        q_avg_info = f_vent * (int(z.n_scoops) * hp_sc + avg_v * hp_volq)
        cap_z = float(abet["vent_cap_max_m3_min_by_zone"].get(zid, 0.0))
        alert_z = bool(abet["vent_alert_by_zone"].get(zid, False))
        usage_z = float(abet["vent_cap_usage_pct_by_zone"].get(zid, 0.0))
        rows.append({
            "Zona / Circuito": f"{z.zone_name} ({zid})",
            "Scooptrams (und)": int(z.n_scoops),
            "Pico Volq. Interior Mina (und)": peak_v,
            "Prom. Volq. Interior Mina (info)": round(avg_v, 2),
            "Q Pico Requerido (m³/min)": round(q_peak, 2),
            "Q Pico Requerido (CFM)": round(q_peak_cfm, 1),
            "Q Prom. Interior (m³/min, info)": round(q_avg_info, 2),
            "Tope Circuito (m³/min)": round(cap_z, 2) if cap_z > 0 else "Sin tope",
            "Uso Tope (%)": f"{usage_z:.1f}%" if cap_z > 0 else "-",
            "Estado Circuito": "ALERTA" if alert_z else ("OK" if cap_z > 0 else "Sin tope"),
        })
    return pd.DataFrame(rows)


def _build_full_excel_report(
    cfg: SimulationConfig,
    dash_res: Optional[Dict[str, object]],
    opt_res: Optional[Dict[str, object]],
    comp_res: Optional[Dict[str, object]],
    bench_res: Optional[Dict[str, object]],
    resil_res: Optional[Dict[str, object]],
    abet_summary: Dict[str, object],
) -> bytes:
    """Empaqueta en un solo Excel ejecutivo todos los resultados calculados."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pol_code = normalize_policy(cfg.global_cfg.politica_despacho)
        nom_hr_rep = cfg.global_cfg.sim_time_sec / 3600.0
        real_hr_rep = nom_hr_rep
        total_opex_rep = 0.0
        fleet_op_rep = 0.0
        truck_op_rep = 0.0
        truck_wait_rep = 0.0
        scoop_op_rep = 0.0
        scoop_idle_rep = 0.0
        net_rev_rep = 0.0
        unit_cost_rep = 0.0

        if dash_res is not None:
            mc_rep = dash_res.get("mc_res")
            kpi_rep = dash_res.get("kpi")
            if mc_rep is not None and "effective_horizon_hr" in mc_rep:
                real_hr_rep = float(mc_rep["effective_horizon_hr"]["mean"])
                total_opex_rep = float(mc_rep["total_opex_usd"]["mean"])
                fleet_op_rep = float(mc_rep["fleet_op_cost_usd"]["mean"])
                net_rev_rep = float(mc_rep["net_revenue_usd"]["mean"])
                unit_cost_rep = float(mc_rep["unit_cost_usd_tn"]["mean"])
                df_mc_r = mc_rep["replicas_df"]
                truck_op_rep = float(df_mc_r["truck_op_cost_usd"].mean())
                truck_wait_rep = float(df_mc_r["truck_wait_cost_usd"].mean())
                scoop_op_rep = float(df_mc_r["scoop_op_cost_usd"].mean())
                scoop_idle_rep = float(df_mc_r["scoop_idle_cost_usd"].mean())
            elif kpi_rep is not None:
                real_hr_rep = float(kpi_rep["effective_horizon_sec"]) / 3600.0
                total_opex_rep = float(kpi_rep.get("total_opex_usd", 0.0))
                fleet_op_rep = float(kpi_rep.get("fleet_op_cost_usd", 0.0))
                truck_op_rep = float(kpi_rep.get("truck_op_cost_usd", 0.0))
                truck_wait_rep = float(kpi_rep.get("truck_wait_cost_usd", 0.0))
                scoop_op_rep = float(kpi_rep.get("scoop_op_cost_usd", 0.0))
                scoop_idle_rep = float(kpi_rep.get("scoop_idle_cost_usd", 0.0))
                net_rev_rep = float(kpi_rep.get("net_revenue_usd", 0.0))
                unit_cost_rep = float(kpi_rep.get("unit_cost_usd_tn", 0.0))

        resumen_rows = [
            ("Esquema de Rutas", "9 parámetros anclados en bocamina (desvinculación post-descarga)"),
            ("Modelo de Costos", "Por Estado del Equipo (HM Productivas vs. Standby/Cola)"),
            ("Tarifa Operativa Volquete (US$/HM)", cfg.global_cfg.costo_op_volquete_hr),
            ("Tarifa Standby / Cola Volquete (US$/h)", cfg.global_cfg.costo_cola_volquete_hr),
            ("Tarifa Operativa Scooptram (US$/HM)", cfg.global_cfg.costo_op_scoop_hr),
            ("Tarifa Standby / Improd. Scooptram (US$/h)", cfg.global_cfg.costo_improd_scoop_hr),
            ("Disponibilidad Mecánica Volquetes Dm", cfg.global_cfg.dm_volquete),
            ("Costo Operativo Volquetes HM (US$/gdia)", round(truck_op_rep, 2)),
            ("Costo Standby / Cola Volquetes (US$/gdia)", round(truck_wait_rep, 2)),
            ("Costo Total Flota Volquetes (US$/gdia)", round(fleet_op_rep, 2)),
            ("Costo Operativo Scooptrams HM (US$/gdia)", round(scoop_op_rep, 2)),
            ("Costo Standby Scooptrams (US$/gdia)", round(scoop_idle_rep, 2)),
            ("Costo Operativo Total (US$/gdia)", round(total_opex_rep, 2)),
            ("Costo Unitario (US$/t)", round(unit_cost_rep, 3)),
            ("Ingreso Neto (US$/gdia)", round(net_rev_rep, 2)),
            ("Política de Despacho", _policy_fmt(pol_code)),
            ("Semilla Base (seed)", cfg.global_cfg.semilla),
            ("Modo de Parada", cfg.global_cfg.modo_parada),
            ("Duración Nominal de Guardia (h)", round(nom_hr_rep, 2)),
            ("Duración Real Efectiva (h)", round(real_hr_rep, 3)),
            ("Meta de Producción (t/gdia, suma de cuotas)", cfg.global_cfg.meta_mineral_gdia),
            ("Guardias/Año", cfg.global_cfg.guardias_anio),
            ("Zonas Activas", len([z for z in cfg.zones if z.active])),
            ("Flota Total Volquetes", sum(z.n_trucks for z in cfg.zones if z.active)),
            ("Caudal Incremental Requerido Total ΣQ_pico (m3/min)", abet_summary.get("vent_q_peak_total_m3_min", 0.0)),
            ("Caudal Incremental Requerido Total ΣQ_pico (CFM)", abet_summary.get("vent_q_peak_total_cfm", 0.0)),
            ("Alerta de Ventilación en algún circuito", "SÍ" if abet_summary.get("vent_any_alert") else "NO"),
            ("Combustible Desperdiciado Colas Volq. (L/gdia)", abet_summary.get("diesel_wasted_wait_l_shift", 0.0)),
            ("Combustible Desperdiciado Colas Volq. (L/año)", abet_summary.get("diesel_wasted_wait_l_year", 0.0)),
            ("Costo Combustible Desperdiciado Colas (US$/gdia)", abet_summary.get("diesel_wasted_wait_cost_usd_shift", 0.0)),
            ("Costo Combustible Desperdiciado Colas (US$/año)", abet_summary.get("diesel_wasted_wait_cost_usd_year", 0.0)),
            ("CO2 Desperdiciado/Evitable Colas Volq. (kg/gdia)", abet_summary.get("co2_wasted_wait_kg_shift", 0.0)),
            ("CO2 Desperdiciado/Evitable Colas Volq. (t/año)", abet_summary.get("co2_wasted_wait_t_year", 0.0)),
            ("Diésel Ahorrado vs Base (L/año)", abet_summary.get("diesel_saved_l_year", 0.0)),
            ("CO2 Evitado vs Base (t/año)", abet_summary.get("co2_avoided_t_year", 0.0)),
        ]
        pd.DataFrame(resumen_rows, columns=["Indicador", "Valor"]).to_excel(
            writer, sheet_name="1_Resumen_ABET", index=False
        )

        # Desglose de ventilación por zona/circuito
        _build_ventilation_zone_df(cfg, abet_summary).to_excel(
            writer, sheet_name="1B_Ventilacion_Zonas", index=False
        )

        if dash_res is not None:
            dash_res["df_econ"].to_excel(writer, sheet_name="2_Economia_Escenario", index=False)
            dash_res["df_stations"].to_excel(writer, sheet_name="3_Estaciones_Colas_DES", index=False)
            if dash_res.get("mc_res") is not None:
                dash_res["mc_res"]["replicas_df"].to_excel(writer, sheet_name="4_Replicas_MC", index=False)

        if opt_res is not None:
            opt_res["fleet_curve_df"].to_excel(writer, sheet_name="5_Curva_SimOpt", index=False)
            opt_res["all_candidates_df"].to_excel(writer, sheet_name="6_Candidatos_SimOpt", index=False)

        if comp_res is not None:
            comp_res["comparison_table"].to_excel(writer, sheet_name="7_Fijo_vs_Dinamico", index=False)

        if bench_res is not None:
            bench_res["summary_df"].to_excel(writer, sheet_name="8_Benchmark_Politicas", index=False)

        if resil_res is not None:
            resil_res["summary_df"].to_excel(writer, sheet_name="9_Resiliencia_Fallas", index=False)

        for name in writer.sheets:
            ws = writer.sheets[name]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 44)

    return buf.getvalue()


# ==============================================================================
# INICIO DE LA APP Y BARRA LATERAL
# ==============================================================================
_init_state()
cfg: SimulationConfig = st.session_state.config
g = cfg.global_cfg
g.politica_despacho = normalize_policy(g.politica_despacho)
g.avoid_down_zones = True
_sync_meta_from_zones(cfg)
v = st.session_state.cfg_version

st.title("⛏️ SubHaul-Opt — Simulación-Optimización Estocástica (DES)")
st.caption(
    "Feria ABET UNI 2026-II · FIGMM · Ingeniería de Minas (MI753-A) | "
    "Despacho Predictivo por ETA, Dimensionamiento de Flotas, Resiliencia y Criterios ABET"
)

with st.sidebar:
    st.header("⚙️ Panel de Control")

    # 1. Botón Caso Base Validado (Huancapetí — leído directo desde motor_des)
    st.subheader("1. Caso Base Validado")
    preset_choice = st.radio(
        "Escenario Huancapetí",
        options=[
            "Expansión 15 Volq. (6-2-7)",
            "Expansión 15 Volq. Asig. E (6-1-8)",
            "Validación Histórica 12 Volq. (4-2-6)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    if st.button("🔄 Cargar Caso Base Validado", type="primary", use_container_width=True):
        if "12 Volq." in preset_choice:
            new_cfg = build_huancapeti_base_config(
                n_her=4, n_car=2, n_cot=6, policy="FIJA", prob_mode="GPSS_WORLD_VAL", seed=14
            )
        elif "Asig. E" in preset_choice:
            new_cfg = build_huancapeti_base_config(
                n_her=6, n_car=1, n_cot=8, policy="DIN_ETA_RITMO", prob_mode="FORECAST", seed=25
            )
        else:
            new_cfg = build_huancapeti_base_config(
                n_her=6, n_car=2, n_cot=7, policy="DIN_ETA_RITMO", prob_mode="FORECAST", seed=25
            )
        _load_new_config(new_cfg)
        st.rerun()

    # 2. Importación / Exportación de Plantilla Excel
    st.subheader("2. Plantilla Excel (N Zonas)")
    uploaded = st.file_uploader("Cargar plantilla (.xlsx)", type=["xlsx"], key=f"upl_{v}")
    if uploaded is not None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                p = Path(tmpdir) / "input.xlsx"
                p.write_bytes(uploaded.getvalue())
                is_valid, err_msg = _validate_uploaded_excel(p)
                if not is_valid:
                    st.error(err_msg)
                else:
                    loaded_cfg = load_config_from_excel(p)
                    route_errs = _validate_zones_routes(loaded_cfg.zones, loaded_cfg.distributions)
                    if route_errs:
                        st.error("La plantilla contiene referencias de ruta no válidas:\n- " + "\n- ".join(route_errs))
                    else:
                        _load_new_config(loaded_cfg)
                        st.success(f"Plantilla cargada ({len(loaded_cfg.zones)} zonas).")
                        st.rerun()
        except Exception as exc:
            st.error(f"Error al procesar la plantilla Excel: {exc}")

    st.download_button(
        label="📥 Descargar Plantilla Excel Actual",
        data=_template_bytes(cfg),
        file_name="plantilla_subhaul_opt.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.divider()

    # 3. Parámetros Globales de Operación e Infraestructura
    with st.expander("🛠️ Operación e Infraestructura", expanded=True):
        shift_hrs = st.number_input("Duración de guardia (h)", 1.0, 24.0, float(g.sim_time_sec / 3600.0), 0.5, key=f"hrs_{v}")
        g.sim_time_sec = shift_hrs * 3600.0
        st.caption(
            f"📌 Meta diaria total calculada (suma de cuotas activas): **{g.meta_mineral_gdia:,.0f} t/gdia** "
            "(editable por zona en el *Editor en Vivo de Zonas*)."
        )
        g.guardias_anio = int(st.number_input("Guardias operativas/año", 1, 1095, int(g.guardias_anio), 10, key=f"gdias_{v}"))
        g.capacidad_volquete_tn = st.number_input("Capacidad nominal volquete (t)", 5.0, 120.0, float(g.capacidad_volquete_tn), 1.0, key=f"capv_{v}")
        g.politica_despacho = st.selectbox(
            "Política de despacho",
            POLICY_OPTIONS,
            index=POLICY_OPTIONS.index(g.politica_despacho) if g.politica_despacho in POLICY_OPTIONS else 0,
            format_func=_policy_fmt,
            help=(
                "Tras descargar en cancha o desmontera, el volquete se desvincula de su zona de origen. "
                "Las 3 políticas predictivas eligen el nuevo destino usando cuota proyectada y ETA (retorno + bajada)."
            ),
            key=f"pol_{v}",
        )
        g.modo_parada = st.selectbox(
            "Corte de fin de guardia",
            STOP_OPTIONS,
            index=STOP_OPTIONS.index(g.modo_parada) if g.modo_parada in STOP_OPTIONS else 0,
            help="HARD: corte exacto al finalizar la guardia nominal. SOFT: volquetes en tránsito culminan su descarga.",
            key=f"stop_{v}",
        )
        g.num_balanzas = int(st.number_input("N° balanzas superficie (0 = desactivar)", 0, 10, int(g.num_balanzas), 1, key=f"bal_{v}"))
        g.cap_descarga_mineral = int(st.number_input("Bahías descarga mineral (0 = sin cola)", 0, 10, int(g.cap_descarga_mineral), 1, key=f"dore_{v}"))
        g.cap_descarga_desmonte = int(st.number_input("Bahías descarga desmonte (0 = sin cola)", 0, 10, int(g.cap_descarga_desmonte), 1, key=f"dwst_{v}"))

    # 4. Montecarlo y Estadística (control único y funcional de semilla)
    with st.expander("🎲 Montecarlo y Estadística", expanded=True):
        g.num_replicas = int(
            st.number_input(
                "Réplicas Montecarlo",
                min_value=2,
                max_value=300,
                value=min(300, max(2, int(g.num_replicas))),
                step=5,
                key=f"nrep_{v}",
                help="Controla exclusivamente el botón de simulación Montecarlo en la Pestaña 1 (Dashboard Ejecutivo).",
            )
        )
        seed_widget_key = f"seed_{v}"
        if seed_widget_key not in st.session_state:
            st.session_state[seed_widget_key] = int(g.semilla)

        sc1, sc2 = st.columns([1.6, 1.0])
        seed_input = int(
            sc1.number_input(
                "Semilla base (seed)",
                min_value=0,
                max_value=999999,
                step=1,
                key=seed_widget_key,
                help="Control único de semilla reproducible para 1 guardia, Montecarlo (CRN), Sim-Opt, Estadística y Simulación 2D.",
            )
        )
        with sc2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Nueva", use_container_width=True, key=f"rand_seed_{v}"):
                new_sd = random.randint(1, 999999)
                g.semilla = new_sd
                st.session_state.last_seed = new_sd
                st.session_state.cfg_version += 1
                st.session_state.dash_res = None
                st.session_state.opt_res = None
                st.session_state.comp_res = None
                st.session_state.bench_res = None
                st.session_state.resil_res = None
                st.session_state.sim2d_res = None
                st.rerun()

        g.semilla = seed_input
        if int(st.session_state.last_seed) != seed_input:
            st.session_state.last_seed = seed_input
            st.session_state.dash_res = None
            st.session_state.opt_res = None
            st.session_state.comp_res = None
            st.session_state.bench_res = None
            st.session_state.resil_res = None
            st.session_state.sim2d_res = None

        g.nivel_confianza = st.slider("Nivel de confianza", 0.80, 0.99, float(g.nivel_confianza), 0.01, key=f"conf_{v}")

    # 5. Estructura Económica Unificada por Estado del Equipo (50 / 30 / 160 / 90 US$/h)
    with st.expander("💰 Costos y Economía", expanded=True):
        g.costo_op_volquete_hr = st.number_input(
            "Tarifa Operativa Volquete (US$/HM)",
            min_value=0.0,
            max_value=1000.0,
            value=float(g.costo_op_volquete_hr),
            step=5.0,
            help="Parámetro referencial configurable (defecto: 50 US$/HM). Solo horas máquina productivas (viaje, carguío, pesaje y descarga).",
            key=f"coptr_{v}",
        )
        g.costo_cola_volquete_hr = st.number_input(
            "Tarifa Standby / Cola Volquete (US$/h)",
            min_value=0.0,
            max_value=1000.0,
            value=float(g.costo_cola_volquete_hr),
            step=5.0,
            help="Parámetro referencial configurable (defecto: 30 US$/h). Solo horas en cola/espera (no se superpone con la tarifa operativa).",
            key=f"ctr_{v}",
        )
        g.costo_op_scoop_hr = st.number_input(
            "Tarifa Operativa Scooptram (US$/HM)",
            min_value=0.0,
            max_value=2000.0,
            value=float(g.costo_op_scoop_hr),
            step=10.0,
            help="Parámetro referencial configurable (defecto: 160 US$/HM). Solo horas máquina efectivas de carguío.",
            key=f"copsc_{v}",
        )
        g.costo_improd_scoop_hr = st.number_input(
            "Tarifa Standby / Improd. Scooptram (US$/h)",
            min_value=0.0,
            max_value=2000.0,
            value=float(g.costo_improd_scoop_hr),
            step=5.0,
            help="Parámetro referencial configurable (defecto: 90 US$/h). Solo horas en standby esperando volquete (las fallas mecánicas no se cobran).",
            key=f"csc_{v}",
        )

    # 6. Criterios ABET (Ventilación D.S. 023-2017-EM y Combustible/CO2 en colas)
    with st.expander("🌬️ ABET: Ventilación (D.S. 023) y CO2", expanded=False):
        g.hp_volquete = st.number_input(
            "Potencia volquete (HP)",
            min_value=50.0,
            max_value=1500.0,
            value=float(g.hp_volquete),
            step=10.0,
            help="Parámetro referencial configurable. Potencia nominal del volquete usada en el cálculo de caudal pico simultáneo por zona.",
            key=f"hpv_{v}",
        )
        g.dm_volquete = st.number_input(
            "Disponibilidad mecánica volquete (Dm)",
            min_value=0.500,
            max_value=1.000,
            value=float(g.dm_volquete),
            step=0.001,
            format="%.3f",
            help=(
                "Dato de tesis: Dm = 0.933 (93.33%), promedio ponderado Tabla 2 "
                "(8 volquetes Volvo a 93.6% y 4 Mercedes a 92.8%). "
                "Se usa únicamente en métricas de disponibilidad/flota y NUNCA descuenta el caudal pico Q_pico,zona."
            ),
            key=f"dmv_{v}",
        )
        g.hp_scoop = st.number_input(
            "Potencia scooptram (HP)",
            min_value=50.0,
            max_value=1000.0,
            value=float(g.hp_scoop),
            step=10.0,
            help="Parámetro referencial configurable. Potencia nominal del scooptram en cámara usada para el caudal incremental por zona.",
            key=f"hps_{v}",
        )
        g.dm_scoop = st.number_input(
            "Disponibilidad mecánica scooptram (Dm)",
            min_value=0.500,
            max_value=1.000,
            value=float(g.dm_scoop),
            step=0.005,
            format="%.3f",
            help="Parámetro referencial configurable de disponibilidad de carguío. No descuenta el caudal pico Q_pico,zona.",
            key=f"dms_{v}",
        )
        g.factor_vent_m3_min_hp = st.number_input(
            "Factor normativo (m³/min/HP)",
            min_value=1.0,
            max_value=10.0,
            value=float(g.factor_vent_m3_min_hp),
            step=0.5,
            help="Parámetro normativo/físico: 3.00 m³/min/HP exigido por el D.S. 023-2017-EM (Art. 252 y Anexo 38) para equipos diésel en interior mina.",
            key=f"fvent_{v}",
        )
        g.diesel_ralenti_l_hr = st.number_input(
            "Consumo diésel ralentí volquete (L/h)",
            min_value=0.5,
            max_value=20.0,
            value=float(g.diesel_ralenti_l_hr),
            step=0.25,
            help="Parámetro referencial configurable. Aplicado exclusivamente a horas en cola de volquetes (excluye tiempo idle de scooptrams).",
            key=f"dies_{v}",
        )
        g.precio_diesel_usd_l = st.number_input(
            "Precio diésel (US$/L)",
            min_value=0.2,
            max_value=5.0,
            value=float(g.precio_diesel_usd_l),
            step=0.05,
            help="Parámetro referencial configurable para valorar el combustible desperdiciado en colas de volquetes.",
            key=f"pdies_{v}",
        )
        g.factor_co2_kg_l = st.number_input(
            "Factor emisión CO2 (kg/L)",
            min_value=1.0,
            max_value=5.0,
            value=float(g.factor_co2_kg_l),
            step=0.01,
            help="Parámetro normativo/físico: 2.68 kg CO₂/L de diésel quemado (factor estándar IPCC / MINAM).",
            key=f"co2_{v}",
        )

    # 7. Contingencia de Falla de Scooptram (What-If)
    st.subheader("3. Falla de Scooptram (What-If)")
    zone_ids_all = [z.zone_id for z in cfg.zones if z.active] or ["HER"]
    fail_active = st.toggle("🚨 Activar parada imprevista en simulación", value=False, key=f"fail_on_{v}")
    fail_zone = st.selectbox("Zona afectada", zone_ids_all, index=min(len(zone_ids_all) - 1, 2), key=f"fail_z_{v}")
    fail_start = st.number_input("Minuto de inicio", 0.0, float(shift_hrs * 60.0), 180.0, 15.0, key=f"fail_st_{v}")
    fail_dur = st.number_input("Duración parada (min)", 5.0, 360.0, 60.0, 15.0, key=f"fail_du_{v}")
    # Desvío ante fallas internalizado siempre en True en las políticas dinámicas
    g.avoid_down_zones = True

    current_breakdown = BreakdownConfig(
        event_id=f"FALLA_{fail_zone}",
        station_key=fail_zone,
        mode="SCHEDULED",
        start_min=float(fail_start),
        duration_min=float(fail_dur),
        units_affected=1,
        active=bool(fail_active),
    )
    cfg.breakdowns = [current_breakdown] if fail_active else []


# ==============================================================================
# EDITOR EN VIVO DE N ZONAS (ESQUEMA CANÓNICO DE 9 RUTAS + CUOTAS -> META DIARIA)
# ==============================================================================
with st.expander(f"🗺️ Editor en Vivo de Zonas de Carguío ({len(cfg.zones)} zonas configuradas — agregar/eliminar filas)", expanded=False):
    st.caption(
        "Las 9 columnas de tiempo de ruta están ancladas en la **bocamina/acceso** de cada zona. "
        "Cada celda acepta claves de distribución separadas por `|` (ej. `T4|T4HKC|T9`) o constantes numéricas "
        "en segundos, incluyendo `0`. Configura `Cuota Mín. (t)` por zona para definir la meta diaria total y "
        "`Tope Vent. Circuito (m³/min)` por bocamina (0 = sin tope)."
    )
    df_zones_in = _zones_to_df(cfg.zones)
    edited_zones_df = st.data_editor(
        df_zones_in,
        num_rows="dynamic",
        use_container_width=True,
        key=f"zones_editor_{v}",
        column_config={
            "zone_id": st.column_config.TextColumn("ID Zona", required=True),
            "zone_name": st.column_config.TextColumn("Nombre"),
            "active": st.column_config.CheckboxColumn("Activa"),
            "n_trucks": st.column_config.NumberColumn("Volquetes", min_value=0, max_value=50, step=1),
            "n_scoops": st.column_config.NumberColumn("Scooptrams", min_value=1, max_value=10, step=1),
            "prob_mineral": st.column_config.NumberColumn("Prob. Mineral", min_value=0.0, max_value=1.0, step=0.01),
            "quota_mineral_tn": st.column_config.NumberColumn(
                "Cuota Mín. (t)",
                min_value=0.0,
                step=50.0,
                help="Cuota objetivo de mineral para esta zona. La suma de las cuotas activas determina automáticamente la Meta de Producción diaria.",
            ),
            "max_quota_mineral_tn": st.column_config.NumberColumn(
                "Cuota Máx. (t, 0=sin tope)",
                min_value=0.0,
                step=50.0,
                help="0 = sin límite superior. Si es >0, el despacho dinámico deja de enviar volquetes al alcanzar este tonelaje proyectado.",
            ),
            "ore_price_usd_tn": st.column_config.NumberColumn("NSR (US$/t)", min_value=0.0, step=1.0),
            "waste_tonnage_dist": st.column_config.TextColumn(
                "Ton. Desmonte (t/dist)",
                help="Tonelaje constante de desmonte (ej. 20.25) o clave de distribución.",
            ),
            "cap_max_ventilacion_m3_min": st.column_config.NumberColumn(
                "Tope Vent. Circuito (m³/min)",
                min_value=0.0,
                max_value=100000.0,
                step=250.0,
                help="Capacidad máxima de aire fresco instalada para esta bocamina/circuito (0 = sin tope).",
            ),
            "dist_inicial": st.column_config.TextColumn(
                "T. Inicial (Estac.→Bocamina)",
                help="Superficie: Estacionamiento → Bocamina (se ejecuta 1 vez al iniciar la guardia).",
            ),
            "dist_bajada_vacio": st.column_config.TextColumn(
                "T. Bajada Vacío (Bocamina→Cámara)",
                help="Interior mina vacío: Bocamina → Cámara de carguío.",
            ),
            "dist_carguio": st.column_config.TextColumn(
                "T. Carguío (Cámara)",
                help="Interior mina: tiempo de carguío del scooptram en cámara.",
            ),
            "dist_subida_cargado": st.column_config.TextColumn(
                "T. Subida Cargado (Cámara→Bocamina)",
                help="Interior mina cargado: Cámara de carguío → Bocamina.",
            ),
            "dist_hacia_balanza": st.column_config.TextColumn(
                "T. Hacia Balanza (Bocamina→Balanza)",
                help="Superficie (mineral cargado): Bocamina → Balanza.",
            ),
            "dist_balanza_a_cancha": st.column_config.TextColumn(
                "T. Balanza→Cancha",
                help="Superficie (mineral cargado): Balanza → Cancha de mineral. Ingresa '0' si no hay balanza.",
            ),
            "dist_hacia_desmontera": st.column_config.TextColumn(
                "T. Hacia Desmontera (Bocamina→Desm.)",
                help="Superficie (desmonte cargado): Bocamina → Desmontera.",
            ),
            "dist_retorno_desde_mineral": st.column_config.TextColumn(
                "T. Retorno Mineral (Cancha→Bocamina)",
                help="Superficie (vacío, exclusivo de zona destino): Cancha de mineral → Bocamina.",
            ),
            "dist_retorno_desde_desmonte": st.column_config.TextColumn(
                "T. Retorno Desmonte (Desm.→Bocamina)",
                help="Superficie (vacío, exclusivo de zona destino): Desmontera → Bocamina.",
            ),
            "entry_start_sec": st.column_config.NumberColumn("Inicio Salida (s)", min_value=0.0, step=60.0),
            "entry_interval_sec": st.column_config.NumberColumn("Intervalo Salida (s)", min_value=0.0, step=30.0),
        },
    )
    cfg.zones = _df_to_zones(edited_zones_df, cfg.zones)
    _sync_meta_from_zones(cfg)

    # Sincroniza el tope de planta por defecto si el usuario modificó cuotas en vivo
    if abs(float(st.session_state.last_meta_gdia) - float(g.meta_mineral_gdia)) > 1e-6:
        st.session_state.last_meta_gdia = float(g.meta_mineral_gdia)
        st.session_state[f"opt_pcap_{v}"] = max(100.0, float(g.meta_mineral_gdia))
        st.session_state.dash_res = None

    st.info(
        f"🎯 **Meta diaria total calculada (suma de cuotas activas): {g.meta_mineral_gdia:,.2f} t/gdia**"
    )

    route_validation_errors = _validate_zones_routes(cfg.zones, cfg.distributions)
    if route_validation_errors:
        st.error(
            "Corrige las referencias de ruta antes de simular:\n- " + "\n- ".join(route_validation_errors)
        )

    # Creador rápido de distribuciones paramétricas (Triangular / Normal / Lognormal)
    st.markdown("**Agregar distribución paramétrica rápida (opcional para nuevas zonas)**")
    c_p1, c_p2, c_p3, c_p4, c_p5, c_p6 = st.columns([1.4, 1.3, 1, 1, 1, 1])
    d_name = c_p1.text_input("Nombre clave", placeholder="ej. T_ZONA4", key=f"dn_{v}")
    d_type = c_p2.selectbox("Tipo", ["TRIANGULAR", "NORMAL", "LOGNORMAL", "CONSTANT"], key=f"dt_{v}")
    d_p1 = c_p3.number_input("Param 1 (min/media)", value=200.0, key=f"dp1_{v}")
    d_p2 = c_p4.number_input("Param 2 (moda/desv)", value=280.0, key=f"dp2_{v}")
    d_p3 = c_p5.number_input("Param 3 (max Tri.)", value=400.0, key=f"dp3_{v}")
    if c_p6.button("➕ Guardar Dist.", use_container_width=True):
        if d_name.strip():
            cfg.distributions[d_name.strip()] = DistributionSpec(
                dist_type=d_type, param1=d_p1, param2=d_p2, param3=d_p3, min_val=0.0
            )
            st.success(f"Distribución `{d_name.strip()}` ({d_type}) registrada.")


if route_validation_errors:
    st.warning("Existen referencias de ruta inválidas en el Editor de Zonas. Corrígelas para ejecutar simulaciones.")
    st.stop()


# ==============================================================================
# PESTAÑAS PRINCIPALES (5 MÓDULOS)
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Dashboard Ejecutivo y Colas",
    "🎯 2. Optimizador Sim-Opt",
    "📈 3. Fijo vs. Dinámico, Benchmark y Resiliencia",
    "🌱 4. Evaluación ABET y Reporte Excel",
    "🎬 5. Simulación 2D",
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: DASHBOARD EJECUTIVO Y COLAS
# ------------------------------------------------------------------------------
with tab1:
    col_run1, col_run2, col_info = st.columns([1.2, 1.5, 3.1])
    run_single = col_run1.button("▶️ Simular 1 Guardia", use_container_width=True)
    run_mc = col_run2.button(
        f"🎲 Simular {g.num_replicas} Réplicas (IC {int(g.nivel_confianza*100)}%)",
        type="primary",
        use_container_width=True,
    )
    active_z = [z for z in cfg.zones if z.active]

    need_dash_refresh = (
        run_single
        or run_mc
        or st.session_state.dash_res is None
        or "df_stations" not in st.session_state.dash_res
        or "vent_q_peak_total_m3_min" not in st.session_state.dash_res["kpi"]
        or not any(
            str(lbl).startswith("Scooptram (")
            for lbl in st.session_state.dash_res["df_stations"]["Estación"]
        )
    )

    if need_dash_refresh:
        eng = SubHaulDESEngine(cfg, seed_override=int(g.semilla))
        kpi_one = eng.run()
        mc_res = run_replications(cfg, n_replicas=g.num_replicas, base_seed=int(g.semilla)) if run_mc else None
        st.session_state.dash_res = {
            "kpi": kpi_one,
            "df_fac": eng.get_facilities_dataframe(),
            "df_q": eng.get_queues_dataframe(),
            "df_stations": eng.get_stations_report_dataframe(),
            "df_sv": eng.get_savevalues_dataframe(),
            "df_econ": eng.get_economic_dataframe(include_extended_costs=True),
            "abet": compute_abet_metrics(cfg, kpi_one),
            "mc_res": mc_res,
        }

    dres = st.session_state.dash_res
    kpi = dres["kpi"]
    mc = dres["mc_res"]

    # Duración nominal vs. duración real efectiva de guardia
    nom_hr = g.sim_time_sec / 3600.0
    if mc is not None and "effective_horizon_hr" in mc:
        real_hr = float(mc["effective_horizon_hr"]["mean"])
        real_hr_ci: Optional[float] = float(mc["effective_horizon_hr"]["ci_half"])
        stop_mode_eff = str(mc["stop_mode"]).strip().upper()
    else:
        real_hr = float(kpi["effective_horizon_sec"]) / 3600.0
        real_hr_ci = None
        stop_mode_eff = str(kpi["stop_mode"]).strip().upper()

    overtime_min = max(0.0, (real_hr - nom_hr) * 60.0)
    if stop_mode_eff == "HARD":
        dur_label = f"Duración real: **{real_hr:.2f} h (corte exacto)**"
    elif real_hr_ci is not None:
        dur_label = (
            f"Duración real: **{real_hr:.2f} ± {real_hr_ci:.2f} h "
            f"(+{overtime_min:.1f} min sobre guardia nominal)**"
        )
    else:
        dur_label = f"Duración real: **{real_hr:.2f} h (+{overtime_min:.1f} min sobre guardia nominal)**"

    col_info.info(
        f"Flota actual: **{sum(z.n_trucks for z in active_z)} volquetes** en **{len(active_z)} zonas** "
        f"| Meta diaria (suma cuotas): **{g.meta_mineral_gdia:,.0f} t/gdia** "
        f"| Política: **{_policy_fmt(g.politica_despacho)}** | Semilla: **{g.semilla}** "
        f"| Parada: **{g.modo_parada}** | {dur_label}"
        + (f" | 🚨 **Falla activa en {fail_zone} ({fail_dur:.0f} min)**" if fail_active else "")
    )

    # KPIs Ejecutivos (grilla 2 filas x 3 columnas)
    if mc is not None:
        ore_val = mc["ore_tn"]["mean"]
        ore_sub = f"Meta: {g.meta_mineral_gdia:,.0f} t | ± {mc['ore_tn']['ci_half']:.2f} t (IC {int(mc['confidence']*100)}%)"
        net_val = mc["net_revenue_usd"]["mean"]
        net_sub = f"± ${mc['net_revenue_usd']['ci_half']:,.2f}"
        opex_val = float(mc["total_opex_usd"]["mean"])
        fleet_op_val = float(mc["fleet_op_cost_usd"]["mean"])
        df_mc_r = mc["replicas_df"]
        scoops_op_val = float((df_mc_r["scoop_op_cost_usd"] + df_mc_r["scoop_idle_cost_usd"]).mean())
        opex_sub = f"Flota Volq.: ${fleet_op_val:,.0f} | Scoops: ${scoops_op_val:,.0f}"
        wait_val = mc["truck_wait_hr"]["mean"]
        fu_pct = float(mc["truck_fu"]["mean"]) * 100.0
        wait_sub = f"Fu Volquetes: {fu_pct:.1f}%"
        avg_util = float(np.mean(list(mc["scoop_util_by_zone_mean"].values())) * 100.0)
        scale_util_pct = float(df_mc_r["scale_utilization"].mean()) * 100.0
        q_vent_m3 = float(mc["abet"]["vent_q_peak_total_m3_min"])
        q_vent_cfm = float(mc["abet"]["vent_q_peak_total_cfm"])
    else:
        ore_val = float(kpi["ore_total_tn"])
        ore_sub = f"Meta: {g.meta_mineral_gdia:,.0f} t ({ore_val - g.meta_mineral_gdia:+.1f} t)"
        net_val = float(kpi["net_revenue_usd"])
        net_sub = f"Anual: ${net_val * g.guardias_anio:,.0f}"
        opex_val = float(kpi["total_opex_usd"])
        fleet_op_val = float(kpi["fleet_op_cost_usd"])
        scoops_op_val = float(kpi["scoop_op_cost_usd"]) + float(kpi["scoop_idle_cost_usd"])
        opex_sub = f"Flota Volq.: ${fleet_op_val:,.0f} | Scoops: ${scoops_op_val:,.0f}"
        wait_val = float(kpi["total_truck_wait_hr"])
        fu_pct = float(kpi["truck_fu"]) * 100.0
        wait_sub = f"Fu Volquetes: {fu_pct:.1f}%"
        avg_util = float(np.mean(list(kpi["scoop_utilization_by_zone"].values())) * 100.0)
        scale_util_pct = float(kpi["scale_utilization"]) * 100.0
        q_vent_m3 = float(dres["abet"]["vent_q_peak_total_m3_min"])
        q_vent_cfm = float(dres["abet"]["vent_q_peak_total_cfm"])

    k1, k2, k3 = st.columns(3)
    k1.metric("Producción Mineral", f"{ore_val:,.2f} t/gdia", ore_sub)
    k2.metric("Ingreso Neto", f"${net_val:,.2f} /gdia", net_sub)
    _render_info_kpi(k3, "Costo Operativo Total (US$/gdia)", f"${opex_val:,.2f} /gdia", opex_sub)

    k4, k5, k6 = st.columns(3)
    _render_info_kpi(k4, "Utilización Prom. Scoops", f"{avg_util:.2f}%", f"Balanza: {scale_util_pct:.1f}%")
    _render_info_kpi(k5, "Tiempo Total en Cola", f"{wait_val:.3f} h/gdia", wait_sub)
    _render_info_kpi(k6, "Ventilación Q (D.S. 023)", f"{q_vent_m3:,.0f} m³/min", f"{q_vent_cfm:,.0f} CFM")

    # Cuadro destacado de Tiempo de Cumplimiento de Meta Diaria
    if g.meta_mineral_gdia > 0.0:
        if mc is None:
            ttt_hr = kpi.get("time_to_target_hr")
            if ttt_hr is not None:
                margen_min = (nom_hr - float(ttt_hr)) * 60.0
                st.success(
                    f"⏱️ Cumplimiento de Meta ({g.meta_mineral_gdia:,.0f} t/gdia) — "
                    f"Política: {_policy_fmt(g.politica_despacho)} (Semilla {kpi['seed']}): "
                    f"meta alcanzada en **{_fmt_hm(float(ttt_hr))}** "
                    f"(margen restante sobre guardia nominal: {margen_min:.0f} min)."
                )
            else:
                deficit_tn = max(0.0, g.meta_mineral_gdia - float(kpi["ore_total_tn"]))
                st.warning(
                    f"⚠️ Cumplimiento de Meta ({g.meta_mineral_gdia:,.0f} t/gdia) — "
                    f"Política: {_policy_fmt(g.politica_despacho)} (Semilla {kpi['seed']}): "
                    f"la meta no fue alcanzada en la guardia (producción final: {kpi['ore_total_tn']:,.2f} t; "
                    f"déficit: {deficit_tn:,.2f} t)."
                )
        else:
            ttt_ci = mc.get("time_to_target_hr")
            if ttt_ci is not None:
                ttt_mean = float(ttt_ci["mean"])
                ttt_ci_min = float(ttt_ci["ci_half"]) * 60.0
                st.success(
                    f"⏱️ Tiempo promedio a Meta ({g.meta_mineral_gdia:,.0f} t/gdia) — "
                    f"Política: {_policy_fmt(g.politica_despacho)} ({mc['n_replicas']} réplicas, semilla base {mc['base_seed']}): "
                    f"**{_fmt_hm(ttt_mean)} ± {ttt_ci_min:.1f} min** "
                    f"(meta alcanzada en {mc['n_replicas_target_met']}/{mc['n_replicas']} réplicas · "
                    f"{mc['prob_target_met_pct']:.1f}%)."
                )
            else:
                st.warning(
                    f"⚠️ Cumplimiento de Meta ({g.meta_mineral_gdia:,.0f} t/gdia) — "
                    f"Política: {_policy_fmt(g.politica_despacho)}: "
                    f"la meta no fue alcanzada en ninguna de las {mc['n_replicas']} réplicas "
                    f"(promedio: {mc['ore_tn']['mean']:,.2f} t/gdia)."
                )

    # Banner automático si es escenario de validación histórica (12 volquetes)
    if kpi["n_trucks_total"] == 12 and g.politica_despacho == "FIJA":
        aprox = (1.0 - abs(1200.0 - kpi["ore_total_tn"]) / 1200.0) * 100.0
        st.success(
            f"✅ **Validación Histórica Huancapetí (12 volquetes):** {kpi['ore_total_tn']:.2f} t/gdia vs. "
            f"1,200.00 t/gdia históricas (**{aprox:.2f}% de precisión**, umbral requerido >97%)."
        )

    # Gráficos Plotly: Producción vs Cuota + Utilización y Colas
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        z_names = [z.zone_name for z in active_z]
        ore_vals = [
            mc["ore_by_zone_mean"][z.zone_id] if mc is not None else kpi["ore_by_zone"][z.zone_id]
            for z in active_z
        ]
        quotas = [z.quota_mineral_tn for z in active_z]
        fig_prod = go.Figure()
        fig_prod.add_bar(name="Producción Mineral (t)", x=z_names, y=ore_vals, marker_color="#1f77b4", text=[f"{val:.1f} t" for val in ore_vals], textposition="outside")
        if any(q > 0 for q in quotas):
            fig_prod.add_bar(name="Cuota Objetivo (t)", x=z_names, y=quotas, marker_color="#ff7f0e", opacity=0.75)
        fig_prod.update_layout(title="Producción de Mineral vs. Cuota por Zona", barmode="group", yaxis_title="Toneladas / guardia", height=360, margin=dict(t=45, b=30))
        st.plotly_chart(fig_prod, use_container_width=True)

    with g_col2:
        df_st = dres["df_stations"]
        fig_q = make_subplots(specs=[[{"secondary_y": True}]])
        fig_q.add_bar(
            name="Utilización (%)", x=df_st["Estación"], y=df_st["Uso (%)"],
            marker_color="#2ca02c", opacity=0.85, secondary_y=False
        )
        fig_q.add_scatter(
            name="Espera Real (s)", x=df_st["Estación"], y=df_st["Espera Real (s)"],
            mode="lines+markers", line=dict(color="#d62728", width=3), secondary_y=True
        )
        fig_q.update_layout(title="Utilización de Equipos vs. Tiempo Real de Espera en Cola", height=360, margin=dict(t=45, b=30))
        fig_q.update_yaxes(title_text="Utilización (%)", secondary_y=False)
        fig_q.update_yaxes(title_text="Espera real en cola [s]", secondary_y=True)
        st.plotly_chart(fig_q, use_container_width=True)

    st.subheader("Evaluación Económica por Zona")
    st.dataframe(dres["df_econ"], use_container_width=True, hide_index=True)

    st.subheader("Reporte Integrado de Estaciones, Colas y Demoras (DES)")
    df_stations_view = dres["df_stations"].copy()
    for opt_col in ("T. Balanza (s)", "T. Desc. Min. (s)", "T. Desc. Desm. (s)", "T. Desc. Total (s)"):
        if opt_col in df_stations_view.columns:
            df_stations_view[opt_col] = df_stations_view[opt_col].apply(
                lambda x: "-" if pd.isna(x) else f"{float(x):.1f}"
            )
    st.dataframe(df_stations_view, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# PESTAÑA 2: OPTIMIZADOR SIM-OPT Y CURVA DE SENSIBILIDAD
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Búsqueda Automática de Flota y Asignación Óptima (Sim-Opt)")
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    f_min = int(oc1.number_input("Flota mínima (N_min)", 1, 50, int(g.flota_min), 1))
    f_max = int(oc2.number_input("Flota máxima (N_max)", 1, 50, int(g.flota_max), 1))
    opt_pol = oc3.selectbox(
        "Política a optimizar",
        POLICY_OPTIONS,
        index=POLICY_OPTIONS.index(g.politica_despacho) if g.politica_despacho in POLICY_OPTIONS else 0,
        format_func=_policy_fmt,
    )
    opt_obj = oc4.selectbox("Función objetivo", OBJ_OPTIONS, index=OBJ_OPTIONS.index(g.funcion_objetivo) if g.funcion_objetivo in OBJ_OPTIONS else 0)
    min_q_pct = oc5.slider("Exigencia mín. cuota zona (%)", 0, 100, 90, 5, help="0% = optimización libre sin cuotas por zona.")

    oc6, oc7, oc8, oc9, oc10 = st.columns([1.0, 1.0, 1.35, 1.35, 1.1])
    rep_scr = int(
        oc6.number_input(
            "Réplicas Screening (Etapa 1)",
            min_value=2,
            max_value=300,
            value=15,
            step=1,
            key="simopt_replicas_screen",
        )
    )
    rep_cnf = int(
        oc7.number_input(
            "Réplicas Confirmación (Etapa 2)",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            key="simopt_replicas_confirm",
        )
    )
    req_meta = oc8.checkbox("Exigir cumplimiento de meta global de mineral", value=True)
    use_plant_cap = oc9.checkbox("Limitar ingreso por capacidad de planta", value=False)
    opt_pcap_key = f"opt_pcap_{v}"
    if opt_pcap_key not in st.session_state:
        st.session_state[opt_pcap_key] = max(100.0, float(g.meta_mineral_gdia))
    plant_cap_tn = float(
        oc10.number_input(
            "Tope de planta (t/gdia)",
            min_value=100.0,
            step=50.0,
            disabled=not use_plant_cap,
            key=opt_pcap_key,
        )
    )

    if st.button("🚀 Ejecutar Optimizador Sim-Opt", type="primary"):
        prog_bar = st.progress(0.0, text="Iniciando barrido Sim-Opt...")
        st.session_state.opt_res = optimize_fleet_and_assignment(
            cfg,
            fleet_min=f_min,
            fleet_max=f_max,
            policy=opt_pol,
            objective=opt_obj,
            n_replicas_screen=rep_scr,
            n_replicas_confirm=rep_cnf,
            top_k_confirm=3,
            require_target_met=req_meta,
            min_zone_quota_pct=float(min_q_pct),
            base_seed=int(g.semilla),
            progress_callback=lambda frac, msg: prog_bar.progress(frac, text=msg),
            use_plant_cap=use_plant_cap,
            plant_cap_tn=plant_cap_tn,
        )
        prog_bar.empty()

    if st.session_state.opt_res is not None:
        ores = st.session_state.opt_res
        df_curve = ores["fleet_curve_df"]
        best = ores["best_overall"]
        best_opex = float(best["total_opex_usd"]["mean"])
        best_fleet_op = float(best["fleet_op_cost_usd"]["mean"])
        best_unit_cost = float(best["unit_cost_usd_tn"]["mean"])

        if ores.get("use_plant_cap"):
            cap_shown = float(ores.get("plant_cap_tn", plant_cap_tn))
            st.info(
                f"Tope de planta activo: {cap_shown:,.0f} t/gdia "
                "(corte exacto en el viaje de cruce, remunerando solo la fracción hasta el tope con el NSR de su zona; "
                "toneladas excedentes aportan $0 al ingreso neto mientras los costos operativos siguen sumando)."
            )

        # KPIs del Óptimo (grilla 2 filas x 3 columnas)
        m1, m2, m3 = st.columns(3)
        _render_info_kpi(m1, "Flota Óptima Prescrita", f"{ores['best_fleet_n']} volquetes", f"Asignación: {ores['best_assignment_str']}")
        _render_info_kpi(m2, "Mínima Flota Factible", f"{ores['min_fleet_meeting_target'] or 'No alcanzada'} volquetes", f"Meta: {g.meta_mineral_gdia:,.0f} t/gdia")
        _render_info_kpi(m3, "Producción Óptima", f"{best['ore_tn']['mean']:,.2f} t/gdia", f"± {best['ore_tn']['ci_half']:.2f} t")

        m4, m5, m6 = st.columns(3)
        _render_info_kpi(m4, "Ingreso Neto Óptimo", f"${best['net_revenue_usd']['mean']:,.2f} /gdia", f"± ${best['net_revenue_usd']['ci_half']:,.2f}")
        _render_info_kpi(
            m5,
            "Costo Operativo Total (US$/gdia)",
            f"${best_opex:,.2f} /gdia",
            f"Flota Volq.: ${best_fleet_op:,.0f} | Costo Unit.: ${best_unit_cost:.2f}/t",
        )
        _render_info_kpi(
            m6,
            "Caudal Incremental Requerido (D.S. 023)",
            f"{best['abet']['vent_q_peak_total_m3_min']:,.0f} m³/min",
            f"{best['abet']['vent_q_peak_total_cfm']:,.0f} CFM",
        )

        if st.button("📌 Aplicar Asignación Óptima al Escenario Actual"):
            new_c = cfg.set_fleet_assignment(ores["best_assignment"])
            new_c.global_cfg.politica_despacho = ores["policy"]
            _load_new_config(new_c)
            st.rerun()

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            fig_sens1 = go.Figure()
            fig_sens1.add_scatter(
                x=df_curve["Flota (N)"],
                y=df_curve["Mineral (t/gdia)"],
                error_y=dict(type="data", array=df_curve["IC95 Mineral (±)"], visible=True),
                mode="lines+markers+text",
                text=df_curve["Asignación"],
                textposition="top center",
                name="Mineral (t/gdia)",
                line=dict(color="#1f77b4", width=3),
            )
            fig_sens1.add_hline(y=g.meta_mineral_gdia, line_dash="dash", line_color="#d62728", annotation_text=f"Meta {g.meta_mineral_gdia:,.0f} t")
            fig_sens1.update_layout(title="Curva de Sensibilidad: Flota vs. Producción Mineral (IC 95%)", xaxis_title="Tamaño de Flota (volquetes)", yaxis_title="Mineral (t/gdia)", height=380)
            st.plotly_chart(fig_sens1, use_container_width=True)

        with p_col2:
            fig_sens2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_sens2.add_scatter(
                x=df_curve["Flota (N)"], y=df_curve["Ingreso Neto (US$/gdia)"],
                mode="lines+markers", name="Ingreso Neto (US$/gdia)",
                line=dict(color="#2ca02c", width=3), secondary_y=False
            )
            if "OPEX Total (US$/gdia)" in df_curve.columns:
                fig_sens2.add_scatter(
                    x=df_curve["Flota (N)"], y=df_curve["OPEX Total (US$/gdia)"],
                    mode="lines+markers", name="Costo Operativo Total (US$/gdia)",
                    line=dict(color="#d62728", width=2.5, dash="dash"), secondary_y=False
                )
            fig_sens2.add_bar(
                x=df_curve["Flota (N)"], y=df_curve["Δ Ingreso Neto/Volq (US$/gdia)"],
                name="Δ Marginal/Volq (US$/gdia)", marker_color="#ff7f0e", opacity=0.55, secondary_y=True
            )
            fig_sens2.update_layout(
                title="Ingreso Neto, Costo Operativo y Contribución Marginal",
                xaxis_title="Tamaño de Flota (volquetes)",
                height=380,
            )
            fig_sens2.update_yaxes(title_text="US$ / guardia", secondary_y=False)
            fig_sens2.update_yaxes(title_text="Δ Ingreso Neto Marginal (US$/gdia)", secondary_y=True)
            st.plotly_chart(fig_sens2, use_container_width=True)

        st.markdown("**Tabla Completa de Curva de Sensibilidad y Análisis Marginal**")
        priority_cols = [
            "Flota (N)",
            "Asignación",
            "Política",
            "Mineral (t/gdia)",
            "IC95 Mineral (±)",
            "Excedente Tope (t)",
            "Tiempo a Meta",
            "Cumple Meta (%)",
            "Cumpl. Mín. Cuota Zona (%)",
            "Ingreso Neto (US$/gdia)",
        ]
        ordered_cols = [c for c in priority_cols if c in df_curve.columns] + [
            c for c in df_curve.columns if c not in priority_cols
        ]
        st.dataframe(df_curve[ordered_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Configura el rango de flota y presiona **Ejecutar Optimizador Sim-Opt** para prescribir la flota óptima.")


# ------------------------------------------------------------------------------
# PESTAÑA 3: CONTRASTE 1 vs 1 + BENCHMARK SIMULTÁNEO (5 POLÍTICAS) + RESILIENCIA
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("A. Contraste Estadístico Formal (1 vs. 1): Despacho Fijo vs. Dinámico")
    cc1, cc2 = st.columns(2)
    comp_mode = cc1.selectbox("Tipo de contraste", ["Flota actual: Fijo vs. Dinámico", "Mejor Fijo vs. Mejor Dinámico (Justo)"])
    default_dyn_idx = DYNAMIC_POLICY_OPTIONS.index(g.politica_despacho) if g.politica_despacho in DYNAMIC_POLICY_OPTIONS else 0
    alt_pol = cc2.selectbox(
        "Política dinámica a contrastar",
        DYNAMIC_POLICY_OPTIONS,
        index=default_dyn_idx,
        format_func=_policy_fmt,
    )

    if "Justo" in comp_mode:
        cr1, cr2, cr3 = st.columns(3)
        comp_screen_3a = int(
            cr1.number_input(
                "Réplicas Screening (3.A)",
                min_value=2,
                max_value=300,
                value=15,
                step=1,
                key="comp_screen_3a",
            )
        )
        comp_confirm_3a = int(
            cr2.number_input(
                "Réplicas Confirmación (3.A)",
                min_value=5,
                max_value=300,
                value=30,
                step=5,
                key="comp_confirm_3a",
            )
        )
        comp_q_pct = cr3.slider("Cuota mín. zona (%) [Modo Justo]", 0, 100, 90, 5, key="comp_q")
    else:
        cr1, _ = st.columns([1, 2])
        comp_reps_3a = int(
            cr1.number_input(
                "Réplicas estadísticas (3.A)",
                min_value=10,
                max_value=300,
                value=30,
                step=5,
                key="comp_reps_3a",
            )
        )

    if st.button("⚖️ Ejecutar Contraste Estadístico (t-Student & Mann-Whitney U)", type="primary"):
        with st.spinner("Ejecutando réplicas pareadas (Common Random Numbers)..."):
            if "Justo" in comp_mode:
                fleet_n = sum(z.n_trucks for z in active_z)
                st.session_state.comp_res = compare_optimal_fixed_vs_dynamic(
                    cfg,
                    fleet_size=fleet_n,
                    policy_dynamic=alt_pol,
                    objective=g.funcion_objetivo,
                    n_replicas_screen=comp_screen_3a,
                    n_replicas_confirm=comp_confirm_3a,
                    min_zone_quota_pct=float(comp_q_pct),
                    base_seed=int(g.semilla),
                )
            else:
                st.session_state.comp_res = compare_policies_statistically(
                    cfg,
                    policy_base="FIJA",
                    policy_alt=alt_pol,
                    n_replicas=comp_reps_3a,
                    base_seed=int(g.semilla),
                )

    if st.session_state.comp_res is not None:
        cres = st.session_state.comp_res
        b1, b2 = st.columns(2)
        _render_info_kpi(b1, "Ganancia Neta / Guardia", f"${cres['delta_net_usd_gdia']:+,.2f}", f"p(t) = {cres['test_net']['t_paired_p']:.5f}")
        _render_info_kpi(b2, "Impacto Anual", f"${cres['delta_net_usd_anio']:+,.2f} /año", f"{g.guardias_anio} guardias (semilla base {cres['base_seed']})")

        b3, b4 = st.columns(2)
        _render_info_kpi(
            b3,
            "Mann-Whitney U (p)",
            f"{cres['test_net']['mannwhitney_p']:.5f}",
            "Significativo (p<0.05)" if cres["test_net"]["mannwhitney_p"] < 0.05 else "No significativo",
        )
        _render_info_kpi(
            b4,
            "CO2 Evitado vs. Fijo (Colas Volq.)",
            f"{cres['abet_comparison']['co2_avoided_t_year']:+.3f} t/año",
            f"Diésel: {cres['abet_comparison']['diesel_saved_l_year']:+.1f} L/año (${cres['abet_comparison']['diesel_saved_usd_year']:+,.2f}/año)",
        )

        st.dataframe(cres["comparison_table"], use_container_width=True, hide_index=True)

        # Boxplot comparativo de réplicas
        df_a = cres["result_base"]["replicas_df"].copy()
        df_a["Escenario"] = f"Fija ({cres['result_base']['assignment_str']})"
        df_b = cres["result_alt"]["replicas_df"].copy()
        df_b["Escenario"] = f"{cres['policy_alt_label']} ({cres['result_alt']['assignment_str']})"
        df_comb = pd.concat([df_a, df_b], ignore_index=True)

        bx1, bx2 = st.columns(2)
        with bx1:
            fig_box1 = px.box(df_comb, x="Escenario", y="ore_total_tn", color="Escenario", points="all", title="Distribución de Producción Mineral entre Réplicas")
            fig_box1.update_layout(showlegend=False, yaxis_title="Mineral (t/gdia)", height=350)
            st.plotly_chart(fig_box1, use_container_width=True)
        with bx2:
            fig_box2 = px.box(df_comb, x="Escenario", y="net_revenue_usd", color="Escenario", points="all", title="Distribución de Ingreso Neto entre Réplicas")
            fig_box2.update_layout(showlegend=False, yaxis_title="Ingreso Neto (US$/gdia)", height=350)
            st.plotly_chart(fig_box2, use_container_width=True)

    st.divider()
    st.subheader("B. Benchmark Simultáneo de las 5 Políticas (Common Random Numbers)")
    st.caption(
        "Evalúa en simultáneo la Asignación Fija, la Dinámica Reactiva (Caso Base) y las 3 técnicas predictivas "
        "sobre la flota actual usando exactamente las mismas semillas aleatorias."
    )
    b3_c1, b3_c2 = st.columns([1.2, 2.8])
    bench_reps_3b = int(
        b3_c1.number_input(
            "Réplicas Benchmark CRN (3.B)",
            min_value=10,
            max_value=300,
            value=30,
            step=5,
            key="bench_reps_3b",
        )
    )
    with b3_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_bench_3b = st.button("🏁 Comparar Todas las Políticas en Simultáneo (Benchmark CRN)")

    if run_bench_3b:
        with st.spinner(f"Ejecutando benchmark de 5 políticas × {bench_reps_3b} réplicas..."):
            st.session_state.bench_res = benchmark_all_policies(
                cfg,
                policies=("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
                n_replicas=bench_reps_3b,
                base_seed=int(g.semilla),
            )

    if st.session_state.bench_res is not None:
        df_bench = st.session_state.bench_res["summary_df"]
        st.dataframe(df_bench.drop(columns=["Código"]), use_container_width=True, hide_index=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            fig_b1 = make_subplots(specs=[[{"secondary_y": True}]])
            fig_b1.add_bar(
                name="Ingreso Neto (US$/gdia)",
                x=df_bench["Política"],
                y=df_bench["Ingreso Neto (US$/gdia)"],
                marker_color="#1f77b4",
                opacity=0.85,
                secondary_y=False,
            )
            fig_b1.add_scatter(
                name="Cola Total (h/gdia)",
                x=df_bench["Política"],
                y=df_bench["Cola Total (h/gdia)"],
                mode="lines+markers",
                line=dict(color="#d62728", width=3),
                secondary_y=True,
            )
            fig_b1.update_layout(title="Ingreso Neto vs. Tiempo en Cola por Política", height=370, margin=dict(t=45, b=80))
            fig_b1.update_yaxes(title_text="Ingreso Neto (US$/gdia)", secondary_y=False)
            fig_b1.update_yaxes(title_text="Cola Total (h/gdia)", secondary_y=True)
            st.plotly_chart(fig_b1, use_container_width=True)

        with bc2:
            fig_b2 = px.bar(
                df_bench,
                x="Política",
                y="Cumpl. Mín. Cuota (%)",
                color="Política",
                text="Cumpl. Mín. Cuota (%)",
                title="Equilibrio de Producción: Cumplimiento Mínimo de Cuota por Zona (%)",
            )
            fig_b2.add_hline(y=90.0, line_dash="dash", line_color="#2ca02c", annotation_text="Umbral 90%")
            fig_b2.update_layout(showlegend=False, height=370, margin=dict(t=45, b=80))
            st.plotly_chart(fig_b2, use_container_width=True)

    st.divider()
    st.subheader("C. Prueba de Resiliencia ante Falla Imprevista de Scooptram (PriorityResource)")
    st.caption(
        f"Contingencia configurada en barra lateral: parada en **{fail_zone}** desde minuto **{fail_start:.0f}** "
        f"por **{fail_dur:.0f} min** (desvío dinámico automático siempre activo)."
    )
    r3_c1, r3_c2 = st.columns([1.2, 2.8])
    resil_reps_3c = int(
        r3_c1.number_input(
            "Réplicas Resiliencia (3.C)",
            min_value=10,
            max_value=300,
            value=25,
            step=5,
            key="resil_reps_3c",
        )
    )
    with r3_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_resil_3c = st.button("🚨 Evaluar Resiliencia (Normal vs. Falla por Política)")

    if run_resil_3c:
        with st.spinner(f"Simulando escenarios con y sin falla de scooptram en las 5 políticas ({resil_reps_3c} réplicas)..."):
            bd_eval = BreakdownConfig(
                event_id=f"FALLA_{fail_zone}",
                station_key=fail_zone,
                mode="SCHEDULED",
                start_min=float(fail_start),
                duration_min=float(fail_dur),
                units_affected=1,
                active=True,
            )
            st.session_state.resil_res = evaluate_breakdown_resilience(
                cfg,
                breakdown=bd_eval,
                policies=("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
                n_replicas=resil_reps_3c,
                base_seed=int(g.semilla),
            )

    if st.session_state.resil_res is not None:
        df_resil = st.session_state.resil_res["summary_df"]
        st.dataframe(df_resil.drop(columns=["Código"]), use_container_width=True, hide_index=True)

        rc1, rc2 = st.columns(2)
        with rc1:
            fig_r1 = px.bar(
                df_resil, x="Política", y="Impacto Neto (US$/gdia)", color="Política",
                text="Impacto Neto (US$/gdia)", title="Impacto Económico ante Parada de Scooptram"
            )
            fig_r1.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_r1, use_container_width=True)
        with rc2:
            fig_r2 = px.bar(
                df_resil, x="Política", y="Cola Extra (h/gdia)", color="Política",
                text="Cola Extra (h/gdia)", title="Incremento de Horas en Cola por Falla"
            )
            fig_r2.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig_r2, use_container_width=True)


# ------------------------------------------------------------------------------
# PESTAÑA 4: EVALUACIÓN ABET (VENTILACIÓN POR ZONA + TRÍO COLAS VOLQ.) + EXCEL
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Evaluación Normativa ABET: Ventilación (D.S. 023-2017-EM) y Combustible/CO₂ en Colas")

    if st.session_state.comp_res is not None:
        abet = st.session_state.comp_res["abet_comparison"]
        abet_source = f"Escenario {st.session_state.comp_res['policy_alt_label']} vs. Fijo"
    elif st.session_state.dash_res["mc_res"] is not None:
        abet = st.session_state.dash_res["mc_res"]["abet"]
        abet_source = f"Escenario actual ({g.num_replicas} réplicas, semilla base {g.semilla})"
    else:
        abet = st.session_state.dash_res["abet"]
        abet_source = f"Escenario actual (1 guardia, semilla {g.semilla})"

    st.caption(
        f"Fuente de métricas ABET: **{abet_source}** | "
        f"Disponibilidad mecánica referencial de volquetes ($D_m$): **{abet['dm_volquete']*100:.2f}%** (no descuenta caudal pico)."
    )

    # Alertas por zona/circuito (bocamina)
    zones_with_cap = [z for z in active_z if float(abet["vent_cap_max_m3_min_by_zone"].get(z.zone_id, 0.0)) > 0.0]
    if zones_with_cap:
        for z in zones_with_cap:
            zid = z.zone_id
            q_z = float(abet["vent_q_peak_m3_min_by_zone"][zid])
            cap_z = float(abet["vent_cap_max_m3_min_by_zone"][zid])
            usage_z = float(abet["vent_cap_usage_pct_by_zone"][zid])
            margin_z = float(abet["vent_margin_m3_min_by_zone"][zid])
            if abet["vent_alert_by_zone"][zid]:
                st.error(
                    f"⚠️ **ALERTA DE VENTILACIÓN — Bocamina {z.zone_name} ({zid}):** "
                    f"El caudal pico requerido ({q_z:,.1f} m³/min) supera la capacidad instalada del circuito "
                    f"({cap_z:,.1f} m³/min) en {abs(margin_z):,.1f} m³/min."
                )
            else:
                st.success(
                    f"✅ **Circuito {z.zone_name} ({zid}) dentro de norma:** Uso al {usage_z:.1f}% "
                    f"({q_z:,.1f} / {cap_z:,.1f} m³/min; margen disponible: {margin_z:,.1f} m³/min)."
                )

    st.markdown("### 1. Caudal incremental requerido por zona (flota modelada)")
    v_top1, v_top2 = st.columns(2)
    _render_info_kpi(
        v_top1,
        "Caudal incremental requerido total (Σ Q_pico,zona)",
        f"{abet['vent_q_peak_total_m3_min']:,.1f} m³/min",
        f"{abet['vent_q_peak_total_cfm']:,.0f} CFM (Fu = 1.0, sin factor reductor Dm)",
    )
    _render_info_kpi(
        v_top2,
        "Flota modelada en evaluación",
        f"{abet['n_trucks']} volquetes | {abet['n_scoops_total']} scooptrams",
        f"Dm volquetes (tesis): {abet['dm_volquete']*100:.2f}% | Factor D.S. 023: {g.factor_vent_m3_min_hp:.2f} m³/min/HP",
    )

    df_vent_zones = _build_ventilation_zone_df(cfg, abet)
    st.dataframe(df_vent_zones, use_container_width=True, hide_index=True)

    st.info(
        "📌 **Nota técnica normativa (D.S. 023-2017-EM):** El caudal incremental aquí reportado corresponde al "
        "requerimiento por equipos diésel en simultáneo en interior mina ($F_u = 1.0$, sin promediar ni aplicar $D_m$). "
        "A nivel de frente o cámara de carguío debe garantizarse además la **velocidad mínima de aire (Art. 248)** "
        "y verificar que en ningún instante se superen los **límites permisibles de gases (Anexo 15)** del D.S. 023-2017-EM."
    )

    st.markdown("### 2. Combustible desperdiciado en colas y emisiones de CO₂ (solo volquetes)")
    st.caption(
        "Enfoque acotado estrictamente a las horas en cola de volquetes (tasa de ralentí configurable; "
        "excluye por completo el tiempo no productivo/idle de scooptrams)."
    )

    t1, t2, t3 = st.columns(3)
    _render_info_kpi(
        t1,
        "Combustible desperdiciado en colas",
        f"{abet['diesel_wasted_wait_l_shift']:,.2f} L/gdia",
        f"{abet['diesel_wasted_wait_l_year']:,.1f} L/año ({g.guardias_anio} guardias)",
    )
    _render_info_kpi(
        t2,
        "Costo de combustible desperdiciado",
        f"${abet['diesel_wasted_wait_cost_usd_shift']:,.2f} /gdia",
        f"${abet['diesel_wasted_wait_cost_usd_year']:,.2f} /año (US$ {g.precio_diesel_usd_l:.2f}/L)",
    )
    _render_info_kpi(
        t3,
        "CO₂ desperdiciado / evitable en colas",
        f"{abet['co2_wasted_wait_kg_shift']:,.2f} kg/gdia",
        f"{abet['co2_wasted_wait_t_year']:,.3f} t CO₂/año ({g.factor_co2_kg_l:.2f} kg CO₂/L)",
    )

    df_trio_colas = pd.DataFrame([
        {
            "Indicador (Solo Colas de Volquetes)": "Combustible desperdiciado en colas",
            "Por Guardia": f"{abet['diesel_wasted_wait_l_shift']:,.2f} L/gdia",
            "Anualizado": f"{abet['diesel_wasted_wait_l_year']:,.1f} L/año",
        },
        {
            "Indicador (Solo Colas de Volquetes)": "Costo de combustible desperdiciado",
            "Por Guardia": f"US$ {abet['diesel_wasted_wait_cost_usd_shift']:,.2f} /gdia",
            "Anualizado": f"US$ {abet['diesel_wasted_wait_cost_usd_year']:,.2f} /año",
        },
        {
            "Indicador (Solo Colas de Volquetes)": "CO₂ desperdiciado / evitable en colas",
            "Por Guardia": f"{abet['co2_wasted_wait_kg_shift']:,.2f} kg CO₂/gdia",
            "Anualizado": f"{abet['co2_wasted_wait_t_year']:,.3f} t CO₂/año",
        },
    ])
    st.dataframe(df_trio_colas, use_container_width=True, hide_index=True)

    # Gráficos ABET: Caudal pico por zona vs. promedio informativo + Curva marginal
    vc1, vc2 = st.columns(2)
    with vc1:
        fig_v = go.Figure()
        fig_v.add_bar(
            name="Q Pico Requerido (m³/min)",
            x=df_vent_zones["Zona / Circuito"],
            y=df_vent_zones["Q Pico Requerido (m³/min)"],
            marker_color="#1f77b4",
            text=[f"{val:,.0f}" for val in df_vent_zones["Q Pico Requerido (m³/min)"]],
            textposition="outside",
        )
        fig_v.add_bar(
            name="Q Prom. Interior (info, m³/min)",
            x=df_vent_zones["Zona / Circuito"],
            y=df_vent_zones["Q Prom. Interior (m³/min, info)"],
            marker_color="#94a3b8",
            opacity=0.65,
        )
        fig_v.update_layout(
            title="Caudal Incremental Requerido por Zona (Pico Concurrente vs. Promedio Info)",
            barmode="group",
            yaxis_title="Caudal (m³/min)",
            height=360,
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with vc2:
        if st.session_state.opt_res is not None:
            df_c = st.session_state.opt_res["fleet_curve_df"]
            fig_m = make_subplots(specs=[[{"secondary_y": True}]])
            fig_m.add_bar(
                x=df_c["Flota (N)"],
                y=df_c["Δ Ventilación/Volq (m3/min)"],
                name="Δ Ventilación Q Pico/Volq (m³/min)",
                marker_color="#17becf",
                secondary_y=False,
            )
            fig_m.add_scatter(
                x=df_c["Flota (N)"],
                y=df_c["CO2 Cola (kg/gdia)"],
                mode="lines+markers",
                name="CO₂ Cola Volq. (kg/gdia)",
                line=dict(color="#e377c2", width=3),
                secondary_y=True,
            )
            fig_m.update_layout(
                title="Impacto Marginal de Ventilación Pico y CO₂ en Colas por Tamaño de Flota",
                xaxis_title="Flota (N)",
                height=360,
            )
            fig_m.update_yaxes(title_text="Δ Caudal Pico / Volquete (m³/min)", secondary_y=False)
            fig_m.update_yaxes(title_text="CO₂ en Cola Volq. (kg/gdia)", secondary_y=True)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("Ejecuta el **Optimizador Sim-Opt** (Pestaña 2) para visualizar aquí el caudal pico marginal y el CO₂ en colas por volquete adicional.")

    st.divider()
    st.subheader("📥 Exportar Reporte Ejecutivo Completo (.xlsx)")
    st.write("Descarga todas las tablas generadas en esta sesión (Resumen ABET, Ventilación por Zonas, Economía, Estaciones y Colas DES, Curva Sim-Opt, Contraste Estadístico, Benchmark y Resiliencia) en un único archivo Excel.")
    report_bytes = _build_full_excel_report(
        cfg,
        st.session_state.dash_res,
        st.session_state.opt_res,
        st.session_state.comp_res,
        st.session_state.bench_res,
        st.session_state.resil_res,
        abet,
    )
    st.download_button(
        label="📊 Descargar Reporte Ejecutivo SubHaul-Opt (.xlsx)",
        data=report_bytes,
        file_name="Reporte_Ejecutivo_SubHaul_Opt.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------------------------
# PESTAÑA 5: SIMULACIÓN VISUAL 2D (HTML5 CANVAS)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("🎬 Simulación Visual 2D de Flota y Despacho en Tiempo Real")
    st.caption(
        f"Hereda automáticamente la flota, zonas, cuotas, tarifas, infraestructura, corte de guardia, falla What-If "
        f"y la semilla base (**{g.semilla}**) configurada en *Montecarlo y Estadística*."
    )

    VIEW_MODES_2D = [
        "Vista individual (1 política)",
        "Comparativo lado a lado (Fija vs. Dinámica)",
    ]

    s2d_c1, s2d_c2, s2d_c3 = st.columns([1.5, 1.7, 1.4])
    view_mode_2d = s2d_c1.selectbox(
        "Modo de vista",
        VIEW_MODES_2D,
        index=0,
        key=f"sim2d_mode_{v}",
    )
    is_compare_2d = view_mode_2d == VIEW_MODES_2D[1]

    if is_compare_2d:
        default_dyn_2d_idx = (
            DYNAMIC_POLICY_OPTIONS.index(g.politica_despacho)
            if g.politica_despacho in DYNAMIC_POLICY_OPTIONS
            else 0
        )
        pol_2d = s2d_c2.selectbox(
            "Política dinámica (vs. Fija)",
            DYNAMIC_POLICY_OPTIONS,
            index=default_dyn_2d_idx,
            format_func=_policy_fmt,
            key=f"sim2d_pol_cmp_{v}",
        )
    else:
        pol_2d = s2d_c2.selectbox(
            "Política a animar",
            POLICY_OPTIONS,
            index=POLICY_OPTIONS.index(g.politica_despacho) if g.politica_despacho in POLICY_OPTIONS else 0,
            format_func=_policy_fmt,
            key=f"sim2d_pol_single_{v}",
        )

    seed_2d = int(g.semilla)

    with s2d_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_2d = st.button(
            "🎬 Generar / Actualizar Simulación 2D",
            type="primary",
            use_container_width=True,
            key=f"sim2d_run_{v}",
        )

    if run_2d or st.session_state.sim2d_res is None:
        with st.spinner("Generando trazado de eventos DES para animación 2D..."):
            if is_compare_2d:
                cfg_left = cfg.clone()
                cfg_left.global_cfg.politica_despacho = "FIJA"
                eng_left = SubHaulDESEngine(cfg_left, seed_override=seed_2d, record_trace=True)
                kpi_left = eng_left.run()
                trace_left = eng_left.get_trace_payload()

                cfg_right = cfg.clone()
                cfg_right.global_cfg.politica_despacho = pol_2d
                eng_right = SubHaulDESEngine(cfg_right, seed_override=seed_2d, record_trace=True)
                kpi_right = eng_right.run()
                trace_right = eng_right.get_trace_payload()

                viewer_h = 560
                html_2d = build_2d_comparison_html(trace_left, trace_right, height=viewer_h)
                st.session_state.sim2d_res = {
                    "mode": "COMPARE",
                    "seed": seed_2d,
                    "policy_left": "FIJA",
                    "policy_right": pol_2d,
                    "kpi_left": kpi_left,
                    "kpi_right": kpi_right,
                    "n_redirs_left": len(trace_left.get("redirections", [])),
                    "n_redirs_right": len(trace_right.get("redirections", [])),
                    "html": html_2d,
                    "iframe_height": viewer_h + 32,
                }
            else:
                cfg_2d = cfg.clone()
                cfg_2d.global_cfg.politica_despacho = pol_2d
                eng_2d = SubHaulDESEngine(cfg_2d, seed_override=seed_2d, record_trace=True)
                kpi_2d = eng_2d.run()
                trace_2d = eng_2d.get_trace_payload()

                viewer_h = 640
                html_2d = build_2d_viewer_html(trace_2d, height=viewer_h)
                st.session_state.sim2d_res = {
                    "mode": "SINGLE",
                    "seed": seed_2d,
                    "policy": pol_2d,
                    "kpi": kpi_2d,
                    "n_redirs": len(trace_2d.get("redirections", [])),
                    "html": html_2d,
                    "iframe_height": viewer_h + 8,
                }

    s2d = st.session_state.sim2d_res
    if s2d is not None:
        if s2d["mode"] == "COMPARE":
            kl = s2d["kpi_left"]
            kr = s2d["kpi_right"]
            d_ore = float(kr["ore_total_tn"]) - float(kl["ore_total_tn"])
            d_net = float(kr["net_revenue_usd"]) - float(kl["net_revenue_usd"])
            d_wait = float(kr["total_truck_wait_hr"]) - float(kl["total_truck_wait_hr"])

            m2d_1, m2d_2, m2d_3, m2d_4 = st.columns(4)
            m2d_1.metric(
                "Mineral (Din. vs. Fija)",
                f"{kr['ore_total_tn']:,.2f} t/gdia",
                f"{d_ore:+.2f} t vs. Fija ({kl['ore_total_tn']:,.2f} t)",
            )
            m2d_2.metric(
                "Ingreso Neto (Din. vs. Fija)",
                f"${kr['net_revenue_usd']:,.2f} /gdia",
                f"${d_net:+,.2f} vs. Fija",
            )
            m2d_3.metric(
                "Tiempo en Cola (Din. vs. Fija)",
                f"{kr['total_truck_wait_hr']:.3f} h/gdia",
                f"{d_wait:+.3f} h vs. Fija ({kl['total_truck_wait_hr']:.3f} h)",
                delta_color="inverse",
            )
            _render_info_kpi(
                m2d_4,
                "N° Redirecciones Dinámicas",
                f"{s2d['n_redirs_right']}",
                f"Fija: {s2d['n_redirs_left']} | Semilla: {s2d['seed']}",
            )
        else:
            k2d = s2d["kpi"]
            m2d_1, m2d_2, m2d_3, m2d_4 = st.columns(4)
            _render_info_kpi(
                m2d_1,
                "Mineral Total",
                f"{k2d['ore_total_tn']:,.2f} t/gdia",
                f"Desmonte: {k2d['waste_total_tn']:,.2f} t",
            )
            _render_info_kpi(
                m2d_2,
                "Ingreso Neto",
                f"${k2d['net_revenue_usd']:,.2f} /gdia",
                f"OPEX: ${k2d['total_opex_usd']:,.2f}",
            )
            _render_info_kpi(
                m2d_3,
                "Tiempo Total en Cola",
                f"{k2d['total_truck_wait_hr']:.3f} h/gdia",
                f"Fu Volquetes: {k2d['truck_fu'] * 100:.1f}%",
            )
            _render_info_kpi(
                m2d_4,
                "N° Redirecciones Dinámicas",
                f"{s2d['n_redirs']}",
                f"Semilla: {s2d['seed']} | {_policy_fmt(s2d['policy'])}",
            )

        components.html(s2d["html"], height=int(s2d["iframe_height"]), scrolling=False)