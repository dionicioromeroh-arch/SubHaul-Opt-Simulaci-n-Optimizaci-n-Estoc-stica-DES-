"""
SubHaul-Opt — Interfaz Web Interactiva en Streamlit.
Feria ABET UNI 2026-II | Métodos de Explotación Subterránea (MI753-A)

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from motor_des import (
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


def _config_matches_current_schema(cfg: object) -> bool:
    """Verifica que una configuración en session_state posea el esquema vigente de 9 rutas."""
    zones = getattr(cfg, "zones", None)
    if not zones:
        return False
    return all(all(hasattr(z, col) for col in ROUTE_COLUMNS) for z in zones)


def _init_state() -> None:
    if "config" not in st.session_state or not _config_matches_current_schema(st.session_state.config):
        st.session_state.config = build_huancapeti_base_config(
            n_her=6, n_car=2, n_cot=7, policy="DIN_ETA_RITMO", prob_mode="FORECAST", seed=25
        )
        st.session_state.dash_res = None
        st.session_state.opt_res = None
        st.session_state.comp_res = None
        st.session_state.bench_res = None
        st.session_state.resil_res = None
    if "cfg_version" not in st.session_state:
        st.session_state.cfg_version = 0
    for key in ("dash_res", "opt_res", "comp_res", "bench_res", "resil_res"):
        if key not in st.session_state:
            st.session_state[key] = None


def _load_new_config(cfg: SimulationConfig) -> None:
    """Reemplaza la configuración activa y fuerza refresco de widgets."""
    cfg.global_cfg.politica_despacho = normalize_policy(cfg.global_cfg.politica_despacho)
    st.session_state.config = cfg
    st.session_state.cfg_version += 1
    st.session_state.dash_res = None
    st.session_state.opt_res = None
    st.session_state.comp_res = None
    st.session_state.bench_res = None
    st.session_state.resil_res = None


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
        # 1. Resumen Ejecutivo + ABET
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
            ("Costo Manejo Extra Desmonte (US$/t)", cfg.global_cfg.costo_desmonte_tn),
            ("Costo Fijo/Marginal Volquete (US$/gdia)", cfg.global_cfg.costo_marginal_volquete_gdia),
            ("Costo Operativo Volquetes HM (US$/gdia)", round(truck_op_rep, 2)),
            ("Costo Standby / Cola Volquetes (US$/gdia)", round(truck_wait_rep, 2)),
            ("Costo Total Flota Volquetes (US$/gdia)", round(fleet_op_rep, 2)),
            ("Costo Operativo Scooptrams HM (US$/gdia)", round(scoop_op_rep, 2)),
            ("Costo Standby Scooptrams (US$/gdia)", round(scoop_idle_rep, 2)),
            ("Costo Operativo Total (US$/gdia)", round(total_opex_rep, 2)),
            ("Costo Unitario (US$/t)", round(unit_cost_rep, 3)),
            ("Ingreso Neto (US$/gdia)", round(net_rev_rep, 2)),
            ("Política de Despacho", _policy_fmt(pol_code)),
            ("Modo de Parada", cfg.global_cfg.modo_parada),
            ("Duración Nominal de Guardia (h)", round(nom_hr_rep, 2)),
            ("Duración Real Efectiva (h)", round(real_hr_rep, 3)),
            ("Meta de Mineral (t/gdia)", cfg.global_cfg.meta_mineral_gdia),
            ("Guardias/Año", cfg.global_cfg.guardias_anio),
            ("Zonas Activas", len([z for z in cfg.zones if z.active])),
            ("Flota Total Volquetes", sum(z.n_trucks for z in cfg.zones if z.active)),
            ("Ventilación Requerida Q (m3/min)", abet_summary.get("q_total_m3_min", 0.0)),
            ("Ventilación Requerida Q (CFM)", abet_summary.get("q_total_cfm", 0.0)),
            ("Alerta de Ventilación", "SÍ" if abet_summary.get("vent_alert") else "NO"),
            ("Diésel en Cola (L/gdia)", abet_summary.get("diesel_queue_l_gdia", 0.0)),
            ("CO2 en Cola (kg/gdia)", abet_summary.get("co2_queue_kg_gdia", 0.0)),
            ("CO2 en Cola (t/año)", abet_summary.get("co2_queue_t_anio", 0.0)),
            ("Diésel Ahorrado vs Base (L/año)", abet_summary.get("diesel_saved_l_anio", 0.0)),
            ("CO2 Evitado vs Base (t/año)", abet_summary.get("co2_avoided_t_anio", 0.0)),
        ]
        pd.DataFrame(resumen_rows, columns=["Indicador", "Valor"]).to_excel(
            writer, sheet_name="1_Resumen_ABET", index=False
        )

        # 2. Dashboard actual (Economía, Estaciones y Colas DES, Réplicas MC)
        if dash_res is not None:
            dash_res["df_econ"].to_excel(writer, sheet_name="2_Economia_Escenario", index=False)
            dash_res["df_stations"].to_excel(writer, sheet_name="3_Estaciones_Colas_DES", index=False)
            if dash_res.get("mc_res") is not None:
                dash_res["mc_res"]["replicas_df"].to_excel(writer, sheet_name="4_Replicas_MC", index=False)

        # 3. Optimizador Sim-Opt
        if opt_res is not None:
            opt_res["fleet_curve_df"].to_excel(writer, sheet_name="5_Curva_SimOpt", index=False)
            opt_res["all_candidates_df"].to_excel(writer, sheet_name="6_Candidatos_SimOpt", index=False)

        # 4. Comparativa Estadística Fijo vs Dinámico
        if comp_res is not None:
            comp_res["comparison_table"].to_excel(writer, sheet_name="7_Fijo_vs_Dinamico", index=False)

        # 5. Benchmark simultáneo de todas las políticas
        if bench_res is not None:
            bench_res["summary_df"].to_excel(writer, sheet_name="8_Benchmark_Politicas", index=False)

        # 6. Resiliencia ante fallas
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

    # 3. Parámetros Globales de Operación
    with st.expander("🛠️ Operación e Infraestructura", expanded=True):
        shift_hrs = st.number_input("Duración de guardia (h)", 1.0, 24.0, float(g.sim_time_sec / 3600.0), 0.5, key=f"hrs_{v}")
        g.sim_time_sec = shift_hrs * 3600.0
        g.meta_mineral_gdia = st.number_input("Meta mineral (t/gdia)", 0.0, 20000.0, float(g.meta_mineral_gdia), 50.0, key=f"meta_{v}")
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

    # 4. Estadística y Semilla
    with st.expander("🎲 Montecarlo y Estadística", expanded=False):
        g.num_replicas = int(st.number_input("Réplicas Montecarlo", 2, 200, int(g.num_replicas), 5, key=f"nrep_{v}"))
        g.semilla = int(st.number_input("Semilla base (seed)", 0, 999999, int(g.semilla), 1, key=f"seed_{v}"))
        g.nivel_confianza = st.slider("Nivel de confianza", 0.80, 0.99, float(g.nivel_confianza), 0.01, key=f"conf_{v}")

    # 5. Estructura Económica Unificada por Estado del Equipo (HM vs. Standby/Cola)
    with st.expander("💰 Costos y Economía", expanded=True):
        g.costo_op_volquete_hr = st.number_input(
            "Tarifa Operativa Volquete (US$/HM)",
            min_value=0.0,
            max_value=1000.0,
            value=float(g.costo_op_volquete_hr),
            step=5.0,
            help="Solo horas máquina productivas (viaje, carguío, pesaje y descarga).",
            key=f"coptr_{v}",
        )
        g.costo_cola_volquete_hr = st.number_input(
            "Tarifa Standby / Cola Volquete (US$/h)",
            min_value=0.0,
            max_value=1000.0,
            value=float(g.costo_cola_volquete_hr),
            step=5.0,
            help="Solo horas en cola/espera (no se superpone con la tarifa operativa).",
            key=f"ctr_{v}",
        )
        g.costo_op_scoop_hr = st.number_input(
            "Tarifa Operativa Scooptram (US$/HM)",
            min_value=0.0,
            max_value=2000.0,
            value=float(g.costo_op_scoop_hr),
            step=10.0,
            help="Solo horas máquina efectivas de carguío.",
            key=f"copsc_{v}",
        )
        g.costo_improd_scoop_hr = st.number_input(
            "Tarifa Standby / Improd. Scooptram (US$/h)",
            min_value=0.0,
            max_value=2000.0,
            value=float(g.costo_improd_scoop_hr),
            step=5.0,
            help="Solo horas en standby esperando volquete (las fallas mecánicas no se cobran).",
            key=f"csc_{v}",
        )
        g.costo_desmonte_tn = st.number_input(
            "Costo manejo extra desmonte (US$/t)",
            min_value=0.0,
            max_value=200.0,
            value=float(g.costo_desmonte_tn),
            step=1.0,
            help="Opcional. Dejar en 0.00 si el manejo de desmonte ya está cubierto por las tarifas horarias para no duplicar.",
            key=f"cdesm_{v}",
        )
        g.costo_marginal_volquete_gdia = st.number_input(
            "Costo fijo/marginal volquete (US$/gdia)",
            min_value=0.0,
            max_value=10000.0,
            value=float(g.costo_marginal_volquete_gdia),
            step=50.0,
            help="Opcional. Dejar en 0.00 si la posesión/alquiler ya está incluida en las tarifas horarias para no duplicar.",
            key=f"cmarg_{v}",
        )

    # 6. Criterios ABET (Ventilación y CO2)
    with st.expander("🌬️ ABET: Ventilación (D.S. 023) y CO2", expanded=False):
        g.hp_volquete = st.number_input("Potencia volquete (HP)", 50.0, 1500.0, float(g.hp_volquete), 10.0, key=f"hpv_{v}")
        g.dm_volquete = st.slider("Disponibilidad mecánica volquete (Dm)", 0.50, 1.00, float(g.dm_volquete), 0.01, key=f"dmv_{v}")
        g.hp_scoop = st.number_input("Potencia scooptram (HP)", 50.0, 1000.0, float(g.hp_scoop), 10.0, key=f"hps_{v}")
        g.dm_scoop = st.slider("Disponibilidad mecánica scooptram (Dm)", 0.50, 1.00, float(g.dm_scoop), 0.01, key=f"dms_{v}")
        g.factor_vent_m3_min_hp = st.number_input("Factor normativo (m³/min/HP)", 1.0, 10.0, float(g.factor_vent_m3_min_hp), 0.5, key=f"fvent_{v}")
        g.cap_max_ventilacion_m3_min = st.number_input("Capacidad máx. ventilación mina (m³/min, 0=sin tope)", 0.0, 100000.0, float(g.cap_max_ventilacion_m3_min), 500.0, key=f"cvent_{v}")
        g.diesel_ralenti_l_hr = st.number_input("Consumo diésel ralentí (L/h)", 0.5, 20.0, float(g.diesel_ralenti_l_hr), 0.25, key=f"dies_{v}")
        g.precio_diesel_usd_l = st.number_input("Precio diésel (US$/L)", 0.2, 5.0, float(g.precio_diesel_usd_l), 0.05, key=f"pdies_{v}")
        g.factor_co2_kg_l = st.number_input("Factor emisión CO2 (kg/L)", 1.0, 5.0, float(g.factor_co2_kg_l), 0.01, key=f"co2_{v}")

    # 7. Contingencia de Falla de Scooptram (What-If)
    st.subheader("3. Falla de Scooptram (What-If)")
    zone_ids_all = [z.zone_id for z in cfg.zones if z.active] or ["HER"]
    fail_active = st.toggle("🚨 Activar parada imprevista en simulación", value=False, key=f"fail_on_{v}")
    fail_zone = st.selectbox("Zona afectada", zone_ids_all, index=min(len(zone_ids_all) - 1, 2), key=f"fail_z_{v}")
    fail_start = st.number_input("Minuto de inicio", 0.0, float(shift_hrs * 60.0), 180.0, 15.0, key=f"fail_st_{v}")
    fail_dur = st.number_input("Duración parada (min)", 5.0, 360.0, 60.0, 15.0, key=f"fail_du_{v}")
    g.avoid_down_zones = st.checkbox("Desviar flota fuera de zona caída (Dinámico)", value=bool(g.avoid_down_zones), key=f"avd_{v}")

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
# EDITOR EN VIVO DE N ZONAS (ESQUEMA CANÓNICO DE 9 RUTAS ANCLADAS EN BOCAMINA)
# ==============================================================================
with st.expander(f"🗺️ Editor en Vivo de Zonas de Carguío ({len(cfg.zones)} zonas configuradas — agregar/eliminar filas)", expanded=False):
    st.caption(
        "Las 9 columnas de tiempo de ruta están ancladas en la **bocamina/acceso** de cada zona. "
        "Cada celda acepta claves de distribución separadas por `|` (ej. `T4|T4HKC|T9`) o constantes numéricas "
        "en segundos, incluyendo `0` (ej. `0` en `dist_balanza_a_cancha` para minas sin balanza). "
        "Tras descargar en cancha o desmontera, el volquete se desvincula de su zona de origen y retorna directo "
        "usando el retorno de su nueva zona destino."
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
            "quota_mineral_tn": st.column_config.NumberColumn("Cuota Mín. (t)", min_value=0.0, step=50.0),
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
            "dist_inicial": st.column_config.TextColumn(
                "T. Inicial (Estac.→Bocamina)",
                help="Estacionamiento → Bocamina (se ejecuta 1 vez al iniciar la guardia). Claves con '|' o segundos (≥ 0).",
            ),
            "dist_bajada_vacio": st.column_config.TextColumn(
                "T. Bajada Vacío (Bocamina→Cámara)",
                help="Interior mina vacío: Bocamina → Cámara de carguío. Claves con '|' o segundos (≥ 0).",
            ),
            "dist_carguio": st.column_config.TextColumn(
                "T. Carguío (Cámara)",
                help="Tiempo de carguío del scooptram en cámara. Clave de distribución, cadena '|' o segundos (≥ 0).",
            ),
            "dist_subida_cargado": st.column_config.TextColumn(
                "T. Subida Cargado (Cámara→Bocamina)",
                help="Interior mina cargado: Cámara de carguío → Bocamina. Claves con '|' o segundos (≥ 0).",
            ),
            "dist_hacia_balanza": st.column_config.TextColumn(
                "T. Hacia Balanza (Bocamina→Balanza)",
                help="Rama mineral cargado: Bocamina → Balanza (o directo a cancha si no hay balanza). Claves con '|' o segundos (≥ 0).",
            ),
            "dist_balanza_a_cancha": st.column_config.TextColumn(
                "T. Balanza→Cancha",
                help="Rama mineral cargado: Balanza → Cancha de mineral. Ingresa '0' si no hay balanza.",
            ),
            "dist_hacia_desmontera": st.column_config.TextColumn(
                "T. Hacia Desmontera (Bocamina→Desm.)",
                help="Rama desmonte cargado: Bocamina → Desmontera. Claves con '|' o segundos (≥ 0).",
            ),
            "dist_retorno_desde_mineral": st.column_config.TextColumn(
                "T. Retorno Mineral (Cancha→Bocamina)",
                help="Retorno vacío exclusivo de la zona destino: Cancha de mineral → Bocamina. Claves con '|' o segundos (≥ 0).",
            ),
            "dist_retorno_desde_desmonte": st.column_config.TextColumn(
                "T. Retorno Desmonte (Desm.→Bocamina)",
                help="Retorno vacío exclusivo de la zona destino: Desmontera → Bocamina. Claves con '|' o segundos (≥ 0).",
            ),
            "entry_start_sec": st.column_config.NumberColumn("Inicio Salida (s)", min_value=0.0, step=60.0),
            "entry_interval_sec": st.column_config.NumberColumn("Intervalo Salida (s)", min_value=0.0, step=30.0),
        },
    )
    cfg.zones = _df_to_zones(edited_zones_df, cfg.zones)

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
# PESTAÑAS PRINCIPALES (4 MÓDULOS DEL BRIEF)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dashboard Ejecutivo y Colas",
    "🎯 2. Optimizador Sim-Opt",
    "📈 3. Fijo vs. Dinámico, Benchmark y Resiliencia",
    "🌱 4. Evaluación ABET y Reporte Excel",
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: DASHBOARD EJECUTIVO Y COLAS
# ------------------------------------------------------------------------------
with tab1:
    col_run1, col_run2, col_info = st.columns([1.2, 1.4, 2.4])
    run_single = col_run1.button("▶️ Simular 1 Guardia (Semilla)", type="primary", use_container_width=True)
    run_mc = col_run2.button(f"🎲 Simular {g.num_replicas} Réplicas (IC {int(g.nivel_confianza*100)}%)", use_container_width=True)
    active_z = [z for z in cfg.zones if z.active]

    if (
        run_single
        or run_mc
        or st.session_state.dash_res is None
        or "df_stations" not in st.session_state.dash_res
        or "fleet_op_cost_usd" not in st.session_state.dash_res["kpi"]
    ):
        eng = SubHaulDESEngine(cfg, seed_override=g.semilla)
        kpi_one = eng.run()
        mc_res = run_replications(cfg, n_replicas=g.num_replicas, base_seed=1) if run_mc else None
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
        f"| Política: **{_policy_fmt(g.politica_despacho)}** | Costos: **Por Estado (HM / Standby)** "
        f"| Parada: **{g.modo_parada}** | {dur_label}"
        + (f" | 🚨 **Falla activa en {fail_zone} ({fail_dur:.0f} min)**" if fail_active else "")
    )

    # KPIs Ejecutivos (grilla 2 filas x 3 columnas)
    if mc is not None:
        ore_val = mc["ore_tn"]["mean"]
        ore_sub = f"± {mc['ore_tn']['ci_half']:.2f} t (IC {int(mc['confidence']*100)}%)"
        net_val = mc["net_revenue_usd"]["mean"]
        net_sub = f"± ${mc['net_revenue_usd']['ci_half']:,.2f}"
        opex_val = float(mc["total_opex_usd"]["mean"])
        fleet_op_val = float(mc["fleet_op_cost_usd"]["mean"])
        df_mc_r = mc["replicas_df"]
        scoops_op_val = float((df_mc_r["scoop_op_cost_usd"] + df_mc_r["scoop_idle_cost_usd"]).mean())
        opex_sub = f"Flota Volq.: ${fleet_op_val:,.0f} | Scoops: ${scoops_op_val:,.0f}"
        wait_val = mc["truck_wait_hr"]["mean"]
        wait_sub = f"Meta cumplida: {mc['prob_target_met_pct']:.1f}% réplicas"
        avg_util = float(np.mean(list(mc["scoop_util_by_zone_mean"].values())) * 100.0)
        q_vent = mc["abet"]["q_total_m3_min"]
    else:
        ore_val = kpi["ore_total_tn"]
        ore_sub = f"Meta: {g.meta_mineral_gdia:,.0f} t ({ore_val - g.meta_mineral_gdia:+.1f} t)"
        net_val = kpi["net_revenue_usd"]
        net_sub = f"Anual: ${net_val * g.guardias_anio:,.0f}"
        opex_val = float(kpi["total_opex_usd"])
        fleet_op_val = float(kpi["fleet_op_cost_usd"])
        scoops_op_val = float(kpi["scoop_op_cost_usd"]) + float(kpi["scoop_idle_cost_usd"])
        opex_sub = f"Flota Volq.: ${fleet_op_val:,.0f} | Scoops: ${scoops_op_val:,.0f}"
        wait_val = kpi["total_truck_wait_hr"]
        wait_sub = f"Fu Volquetes: {kpi['truck_fu']*100:.1f}%"
        avg_util = float(np.mean(list(kpi["scoop_utilization_by_zone"].values())) * 100.0)
        q_vent = dres["abet"]["q_total_m3_min"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Producción Mineral", f"{ore_val:,.2f} t/gdia", ore_sub)
    k2.metric("Ingreso Neto", f"${net_val:,.2f} /gdia", net_sub)
    k3.metric("Costo Operativo Total (US$/gdia)", f"${opex_val:,.2f} /gdia", opex_sub, delta_color="off")

    k4, k5, k6 = st.columns(3)
    k4.metric("Utilización Prom. Scoops", f"{avg_util:.2f}%", f"Balanza: {kpi['scale_utilization']*100:.1f}%", delta_color="off")
    k5.metric("Tiempo Total en Cola", f"{wait_val:.3f} h/gdia", wait_sub, delta_color="off")
    k6.metric("Ventilación Q (D.S. 023)", f"{q_vent:,.0f} m³/min", f"{q_vent*35.3147:,.0f} CFM", delta_color="off")

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
    rep_scr = int(oc6.number_input("Réplicas Screening (Etapa 1)", 2, 30, 5, 1))
    rep_cnf = int(oc7.number_input("Réplicas Confirmación (Etapa 2)", 5, 60, 15, 5))
    req_meta = oc8.checkbox("Exigir cumplimiento de meta global de mineral", value=True)
    use_plant_cap = oc9.checkbox("Limitar ingreso por capacidad de planta", value=False)
    plant_cap_tn = float(
        oc10.number_input(
            "Tope de planta (t/gdia)",
            min_value=100.0,
            value=max(100.0, float(g.meta_mineral_gdia)),
            step=50.0,
            disabled=not use_plant_cap,
            key=f"opt_pcap_{v}",
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
            top_k_confirm=2,
            require_target_met=req_meta,
            min_zone_quota_pct=float(min_q_pct),
            base_seed=1,
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
                "(corte discreto al último volquete de cruce; toneladas excedentes aportan $0 al ingreso neto "
                "mientras los costos operativos siguen sumando)."
            )

        # KPIs del Óptimo (grilla 2 filas x 3 columnas)
        m1, m2, m3 = st.columns(3)
        m1.metric("Flota Óptima Prescrita", f"{ores['best_fleet_n']} volquetes", f"Asignación: {ores['best_assignment_str']}", delta_color="off")
        m2.metric("Mínima Flota Factible", f"{ores['min_fleet_meeting_target'] or 'No alcanzada'} volquetes", f"Meta: {g.meta_mineral_gdia:,.0f} t/gdia", delta_color="off")
        m3.metric("Producción Óptima", f"{best['ore_tn']['mean']:,.2f} t/gdia", f"± {best['ore_tn']['ci_half']:.2f} t", delta_color="off")

        m4, m5, m6 = st.columns(3)
        m4.metric("Ingreso Neto Óptimo", f"${best['net_revenue_usd']['mean']:,.2f} /gdia", f"± ${best['net_revenue_usd']['ci_half']:,.2f}", delta_color="off")
        m5.metric(
            "Costo Operativo Total (US$/gdia)",
            f"${best_opex:,.2f} /gdia",
            f"Flota Volq.: ${best_fleet_op:,.0f} | Costo Unit.: ${best_unit_cost:.2f}/t",
            delta_color="off",
        )
        m6.metric("Ventilación Q (D.S. 023)", f"{best['abet']['q_total_m3_min']:,.0f} m³/min", f"{best['abet']['q_total_cfm']:,.0f} CFM", delta_color="off")

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
        st.dataframe(df_curve, use_container_width=True, hide_index=True)
    else:
        st.info("Configura el rango de flota y presiona **Ejecutar Optimizador Sim-Opt** para prescribir la flota óptima.")


# ------------------------------------------------------------------------------
# PESTAÑA 3: CONTRASTE 1 vs 1 + BENCHMARK SIMULTÁNEO (5 POLÍTICAS) + RESILIENCIA
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("A. Contraste Estadístico Formal (1 vs. 1): Despacho Fijo vs. Dinámico")
    cc1, cc2, cc3, cc4 = st.columns(4)
    comp_mode = cc1.selectbox("Tipo de contraste", ["Flota actual: Fijo vs. Dinámico", "Mejor Fijo vs. Mejor Dinámico (Justo)"])
    default_dyn_idx = DYNAMIC_POLICY_OPTIONS.index(g.politica_despacho) if g.politica_despacho in DYNAMIC_POLICY_OPTIONS else 0
    alt_pol = cc2.selectbox(
        "Política dinámica a contrastar",
        DYNAMIC_POLICY_OPTIONS,
        index=default_dyn_idx,
        format_func=_policy_fmt,
    )
    comp_reps = int(cc3.number_input("Réplicas estadísticas", 10, 100, int(g.num_replicas), 5))
    comp_q_pct = cc4.slider("Cuota mín. zona (%) [Modo Justo]", 0, 100, 90, 5, key="comp_q")

    if st.button("⚖️ Ejecutar Contraste Estadístico (t-Student & Mann-Whitney U)", type="primary"):
        with st.spinner("Ejecutando réplicas pareadas (Common Random Numbers)..."):
            if "Justo" in comp_mode:
                fleet_n = sum(z.n_trucks for z in active_z)
                st.session_state.comp_res = compare_optimal_fixed_vs_dynamic(
                    cfg,
                    fleet_size=fleet_n,
                    policy_dynamic=alt_pol,
                    objective=g.funcion_objetivo,
                    n_replicas_screen=5,
                    n_replicas_confirm=comp_reps,
                    min_zone_quota_pct=float(comp_q_pct),
                    base_seed=1,
                )
            else:
                st.session_state.comp_res = compare_policies_statistically(
                    cfg,
                    policy_base="FIJA",
                    policy_alt=alt_pol,
                    n_replicas=comp_reps,
                    base_seed=1,
                )

    if st.session_state.comp_res is not None:
        cres = st.session_state.comp_res
        sig = cres["test_net"]["significant_parametric"]
        b1, b2 = st.columns(2)
        b1.metric("Ganancia Neta / Guardia", f"${cres['delta_net_usd_gdia']:+,.2f}", f"p(t) = {cres['test_net']['t_paired_p']:.5f}", delta_color="off")
        b2.metric("Impacto Anual", f"${cres['delta_net_usd_anio']:+,.2f} /año", f"{g.guardias_anio} guardias", delta_color="off")

        b3, b4 = st.columns(2)
        b3.metric("Mann-Whitney U (p)", f"{cres['test_net']['mannwhitney_p']:.5f}", "Significativo (p<0.05)" if sig else "No significativo", delta_color="off")
        b4.metric("CO2 Evitado vs. Fijo", f"{cres['abet_comparison']['co2_avoided_t_anio']:+.2f} t/año", f"Diésel: {cres['abet_comparison']['diesel_saved_l_anio']:+.0f} L/año", delta_color="off")

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
    if st.button("🏁 Comparar Todas las Políticas en Simultáneo (Benchmark CRN)"):
        with st.spinner(f"Ejecutando benchmark de 5 políticas × {comp_reps} réplicas..."):
            st.session_state.bench_res = benchmark_all_policies(
                cfg,
                policies=("FIJA", "DIN_REACTIVA", "DIN_TIEMPO_TOTAL", "DIN_ETA_ESPERA", "DIN_ETA_RITMO"),
                n_replicas=comp_reps,
                base_seed=1,
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
        f"por **{fail_dur:.0f} min**."
    )
    if st.button("🚨 Evaluar Resiliencia (Normal vs. Falla por Política)"):
        with st.spinner("Simulando escenarios con y sin falla de scooptram en las 5 políticas..."):
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
                n_replicas=max(10, min(25, g.num_replicas)),
                base_seed=1,
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
# PESTAÑA 4: EVALUACIÓN ABET (VENTILACIÓN, CO2, COSTOS MARGINALES) + EXCEL
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Evaluación Normativa ABET: Ventilación (D.S. 023-2017-EM) y Emisiones (IPCC)")

    # Usa métricas ABET del contraste si existe, o del dashboard actual
    if st.session_state.comp_res is not None:
        abet = st.session_state.comp_res["abet_comparison"]
        abet_source = f"Escenario {st.session_state.comp_res['policy_alt_label']} vs. Fijo"
    elif st.session_state.dash_res["mc_res"] is not None:
        abet = st.session_state.dash_res["mc_res"]["abet"]
        abet_source = f"Escenario actual ({g.num_replicas} réplicas)"
    else:
        abet = st.session_state.dash_res["abet"]
        abet_source = "Escenario actual (1 corrida)"

    st.caption(f"Fuente de métricas ABET: **{abet_source}**")

    # Alerta de capacidad de ventilación de mina
    if abet["cap_max_vent_m3_min"] > 0:
        if abet["vent_alert"]:
            st.error(
                f"⚠️ **ALERTA DE VENTILACIÓN (D.S. 023-2017-EM):** El caudal requerido "
                f"({abet['q_total_m3_min']:,.1f} m³/min) supera la capacidad instalada de mina "
                f"({abet['cap_max_vent_m3_min']:,.1f} m³/min) en {abs(abet['vent_margin_m3_min']):,.1f} m³/min."
            )
        else:
            st.success(
                f"✅ **Ventilación dentro de norma:** Uso al {abet['vent_usage_pct']:.1f}% de capacidad "
                f"(margen disponible: {abet['vent_margin_m3_min']:,.1f} m³/min)."
            )

    a1, a2 = st.columns(2)
    a1.metric("Caudal Requerido (Q)", f"{abet['q_total_m3_min']:,.1f} m³/min", f"{abet['q_total_cfm']:,.0f} CFM", delta_color="off")
    a2.metric("Diésel en Ralentí (Cola)", f"{abet['diesel_queue_l_gdia']:.2f} L/gdia", f"${abet['diesel_queue_usd_gdia']:.2f} /gdia", delta_color="off")

    a3, a4 = st.columns(2)
    a3.metric("Emisiones CO2 en Cola", f"{abet['co2_queue_kg_gdia']:.2f} kg/gdia", f"{abet['co2_queue_t_anio']:.2f} t CO2/año", delta_color="off")
    a4.metric("CO2 Evitado vs. Base", f"{abet['co2_avoided_t_anio']:+.2f} t/año", f"Ahorro diésel: ${abet['diesel_saved_usd_anio']:+,.0f}/año", delta_color="off")

    # Gráfico de desglose de caudal de ventilación por equipo
    vent_items = [{"Equipo": f"Flota Volquetes ({abet['n_trucks']} und)", "Caudal (m³/min)": abet["q_trucks_m3_min"]}]
    for zid, qz in abet["q_scoops_by_zone_m3_min"].items():
        vent_items.append({"Equipo": f"Scooptram {zid}", "Caudal (m³/min)": qz})
    df_vent = pd.DataFrame(vent_items)

    vc1, vc2 = st.columns(2)
    with vc1:
        fig_v = px.bar(df_vent, x="Equipo", y="Caudal (m³/min)", color="Equipo", text="Caudal (m³/min)", title="Requerimiento de Aire Fresco por Equipo (D.S. 023-2017-EM)")
        if abet["cap_max_vent_m3_min"] > 0:
            fig_v.add_hline(y=abet["cap_max_vent_m3_min"], line_dash="dash", line_color="red", annotation_text="Cap. Máx. Mina")
        fig_v.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_v, use_container_width=True)

    with vc2:
        if st.session_state.opt_res is not None:
            df_c = st.session_state.opt_res["fleet_curve_df"]
            fig_m = make_subplots(specs=[[{"secondary_y": True}]])
            fig_m.add_bar(x=df_c["Flota (N)"], y=df_c["Δ Ventilación/Volq (m3/min)"], name="Δ Ventilación/Volq (m³/min)", marker_color="#17becf", secondary_y=False)
            fig_m.add_scatter(x=df_c["Flota (N)"], y=df_c["CO2 Cola (kg/gdia)"], mode="lines+markers", name="CO2 Cola (kg/gdia)", line=dict(color="#e377c2", width=3), secondary_y=True)
            fig_m.update_layout(title="Impacto Marginal de Ventilación y CO2 por Tamaño de Flota", xaxis_title="Flota (N)", height=350)
            fig_m.update_yaxes(title_text="Δ Caudal / Volquete (m³/min)", secondary_y=False)
            fig_m.update_yaxes(title_text="CO2 en Cola (kg/gdia)", secondary_y=True)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("Ejecuta el **Optimizador Sim-Opt** (Pestaña 2) para visualizar aquí el costo y caudal marginal por volquete adicional.")

    st.divider()
    st.subheader("📥 Exportar Reporte Ejecutivo Completo (.xlsx)")
    st.write("Descarga todas las tablas generadas en esta sesión (Resumen ABET, Economía, Estaciones y Colas DES, Curva Sim-Opt, Contraste Estadístico, Benchmark y Resiliencia) en un único archivo Excel.")
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