"""
SubHaul-Opt — Motor de Simulación de Eventos Discretos (DES) Genérico para N Zonas.
Feria ABET UNI 2026-II | Métodos de Explotación Subterránea (MI753-A)

Características clave:
- Esquema canónico de 9 parámetros de ruta anclados en la bocamina/acceso de cada
  zona (sin suposición de nodo común de retorno en superficie).
- Soporta N zonas dinámicas de carguío, k >= 1 scooptrams por cámara, balanzas
  configurables (o desactivables) y bahías de descarga simultánea.
- Muestreo estocástico híbrido: ojivas empíricas (transformada inversa) y
  distribuciones paramétricas (CONSTANT, TRIANGULAR, NORMAL, LOGNORMAL).
- 5 Políticas de despacho:
    * FIJA             : Asignación Fija (Estática)
    * DIN_ETA_RITMO    : Dinámica: ETA + Ritmo de Cuota (predictiva post-descarga)
    * DIN_ETA_ESPERA   : Dinámica: ETA Menor Espera (predictiva post-descarga)
    * DIN_TIEMPO_TOTAL : Dinámica: Menor Tiempo Total (Viaje + Cola, post-descarga)
    * DIN_REACTIVA     : Dinámica: Reactiva (Caso Base post-descarga)
- Desvío automático incondicional ante zonas completamente detenidas por falla
  mecánica (con fallback de seguridad únicamente si todas las zonas caen).
- Modelo económico unificado por estado del equipo (Opción A: Horas Máquina
  Productivas vs. Horas Standby/Cola, con partición exclusiva de tiempo y cero
  doble conteo; tarifas base 50 / 30 / 160 / 90 US$/h).
- Tope configurable de capacidad de planta remunerada con corte exacto en el
  viaje de cruce y valoración por NSR de la zona de origen.
- Sincronización automática de la meta diaria con la suma de cuotas activas por zona.
- Ventilación normativa D.S. 023-2017-EM calculada sobre el pico simultáneo local
  de volquetes en interior mina por zona (Fu = 1.0, sin factor reductor Dm) y
  topes de capacidad de ventilación por zona/circuito.
- Combustible diésel y emisiones de CO2 desperdiciados estrictamente acotados a
  las horas en cola de volquetes (excluyendo tiempo idle de scooptrams).
- Corte de fin de guardia configurable: HARD (corte exacto en sim_time) y
  SOFT (permite culminar descarga a volquetes en tránsito).
- Contingencias de falla (What-If programada o estocástica MTBF/MTTR) con
  simpy.PriorityResource sin contaminar la utilización productiva.
- Registro opcional de eventos (`record_trace=True`) serializable a JSON para
  animación 2D sin alterar RNG ni KPIs.
- Exportación e importación completa desde plantilla Excel (.xlsx).
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import simpy


# ==============================================================================
# 0. CATÁLOGO OFICIAL DE POLÍTICAS DE DESPACHO (Y COMPATIBILIDAD LEGACY)
# ==============================================================================
POLICY_OPTIONS: List[str] = [
    "DIN_ETA_RITMO",
    "DIN_ETA_ESPERA",
    "DIN_TIEMPO_TOTAL",
    "DIN_REACTIVA",
    "FIJA",
]

POLICY_LABELS: Dict[str, str] = {
    "DIN_ETA_RITMO": "Dinámica: ETA + Ritmo de Cuota",
    "DIN_ETA_ESPERA": "Dinámica: ETA Menor Espera",
    "DIN_TIEMPO_TOTAL": "Dinámica: Menor Tiempo Total (Viaje + Cola)",
    "DIN_REACTIVA": "Dinámica: Reactiva (Caso Base)",
    "FIJA": "Asignación Fija (Estática)",
}

PREDICTIVE_POLICIES = {"DIN_ETA_RITMO", "DIN_ETA_ESPERA", "DIN_TIEMPO_TOTAL"}

_LEGACY_POLICY_MAP: Dict[str, str] = {
    "BASE_HUANCAPETI": "DIN_REACTIVA",
    "DINAMICA": "DIN_ETA_ESPERA",
    "HIBRIDA": "DIN_ETA_RITMO",
}

M3_MIN_TO_CFM: float = 35.3146667


def normalize_policy(policy: str) -> str:
    """Normaliza el código de política, manteniendo compatibilidad con plantillas previas."""
    p = str(policy or "DIN_ETA_RITMO").strip().upper()
    p = _LEGACY_POLICY_MAP.get(p, p)
    return p if p in POLICY_LABELS else "DIN_ETA_RITMO"


def _normalize_route_chain(val: Union[None, str, float, int, Sequence[Union[str, float, int]]]) -> List[str]:
    """
    Normaliza una especificación de ruta a una lista de claves o constantes en texto.
    Soporta claves separadas por '|', constantes numéricas (incluido 0) o secuencias.
    """
    if val is None:
        return []
    if isinstance(val, (int, float)):
        if isinstance(val, float) and math.isnan(val):
            return []
        return [str(val)]
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return []
        return [p.strip() for p in text.split("|") if p.strip()]
    out: List[str] = []
    for item in val:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            if isinstance(item, float) and math.isnan(item):
                continue
            out.append(str(item))
        else:
            text = str(item).strip()
            if not text:
                continue
            out.extend(p.strip() for p in text.split("|") if p.strip())
    return out


# ==============================================================================
# 1. OJIVAS EMPÍRICAS DEL CASO BASE VALIDADO (HUANCAPETÍ)
# ==============================================================================
HUANCAPETI_CDF_FUNCTIONS: Dict[str, List[Tuple[float, float]]] = {
    # Tiempos de traslado (segundos)
    "T1": [(0.0, 35.0), (0.3, 44.0), (0.6, 53.0), (0.7, 62.0), (0.86, 71.0), (0.88, 80.0), (0.94, 89.0), (1.0, 98.0)],
    "T2": [(0.0, 1089.0), (0.08, 1212.43), (0.14, 1335.86), (0.26, 1459.29), (0.46, 1582.71), (0.7, 1706.14), (0.86, 1829.57), (1.0, 1953.0)],
    "T3": [(0.0, 1544.0), (0.02, 1637.71), (0.12, 1731.43), (0.34, 1825.14), (0.56, 1918.86), (0.76, 2012.57), (0.9, 2106.29), (1.0, 2200.0)],
    "T4": [(0.0, 735.0), (0.0967, 789.167), (0.1935, 843.333), (0.4194, 897.5), (0.7419, 951.667), (0.9355, 1005.83), (1.0, 1060.0)],
    "T4HKC": [(0.0, 428.0), (0.1667, 474.429), (0.5833, 520.857), (0.8542, 567.286), (0.8958, 613.714), (0.9583, 660.143), (0.9792, 706.571), (1.0, 753.0)],
    "T9": [(0.0, 737.0), (0.1818, 803.571), (0.4318, 870.143), (0.6818, 936.714), (0.8182, 1003.29), (0.8409, 1069.86), (0.9773, 1136.43), (1.0, 1203.0)],
    "T10": [(0.0, 20.0), (0.0638, 40.7143), (0.4894, 61.4286), (0.8085, 82.1429), (0.9149, 102.857), (0.9575, 123.571), (0.9787, 144.286), (1.0, 165.0)],
    "T11": [(0.0, 418.0), (0.0204, 464.857), (0.0612, 511.714), (0.3061, 558.571), (0.6326, 605.429), (0.7959, 652.286), (0.8776, 699.143), (1.0, 746.0)],
    "T8HKC": [(0.0, 301.0), (0.3529, 336.6), (0.4706, 372.2), (0.5882, 407.8), (0.7941, 443.4), (1.0, 479.0)],
    "T8": [(0.0, 495.0), (0.06, 558.0), (0.18, 621.0), (0.48, 684.0), (0.72, 747.0), (0.94, 810.0), (0.98, 873.0), (1.0, 936.0)],
    "T3K": [(0.0, 502.0), (0.1429, 530.13), (0.4286, 558.286), (0.7143, 586.429), (0.8, 614.571), (0.9428, 642.857), (0.96, 670.857), (1.0, 699.0)],
    "T4K": [(0.0, 595.0), (0.36, 640.625), (0.68, 686.25), (0.76, 731.875), (0.84, 777.5), (0.88, 823.125), (0.89, 868.75), (0.90, 914.375), (1.0, 960.0)],
    "T7": [(0.0, 477.0), (0.06, 588.143), (0.16, 699.286), (0.42, 810.429), (0.66, 921.571), (0.86, 1032.71), (0.9, 1143.86), (1.0, 1255.0)],
    "T1KC": [(0.0, 461.0), (0.04, 568.857), (0.16, 676.714), (0.3, 784.571), (0.44, 892.429), (0.66, 1000.29), (0.84, 1108.14), (1.0, 1216.0)],
    "T5K": [(0.0, 22.0), (0.3333, 33.333), (0.4667, 44.667), (0.8, 56.0), (0.8333, 67.333), (0.9333, 78.667), (1.0, 90.0)],
    "T6K": [(0.0, 13.0), (0.09375, 17.5714), (0.28125, 22.1429), (0.59375, 26.7143), (0.84375, 31.2857), (0.96875, 35.8571), (0.9800, 40.4286), (1.0, 45.0)],
    "T7K": [(0.0, 157.0), (0.16129, 180.0), (0.3223, 203.0), (0.6129, 226.0), (0.8065, 249.0), (0.9355, 272.0), (1.0, 295.0)],
    "T2K": [(0.0, 25.0), (0.233, 30.833), (0.567, 36.667), (0.733, 42.5), (0.8, 48.333), (0.967, 54.167), (1.0, 60.0)],
    "T2C": [(0.0, 628.0), (0.08, 710.286), (0.2, 792.571), (0.32, 874.857), (0.5, 957.143), (0.68, 1039.43), (0.92, 1121.71), (1.0, 1204.0)],
    "T3C": [(0.0, 1000.0), (0.2, 1140.71), (0.625, 1281.43), (0.75, 1422.14), (0.875, 1562.86), (0.925, 1703.57), (0.975, 1844.29), (1.0, 1985.0)],
    "T5": [(0.0, 57.0), (0.075, 84.1429), (0.1, 111.286), (0.125, 138.429), (0.2, 165.571), (0.35, 192.714), (0.85, 219.857), (1.0, 247.0)],
    "T6": [(0.0, 28.0), (0.233, 34.667), (0.6, 41.333), (0.767, 48.0), (0.9, 54.667), (0.9667, 61.333), (1.0, 68.0)],
    # Tiempos de carguío en cámaras (segundos)
    "THER": [(0.0, 165.0), (0.3, 244.286), (0.64, 323.571), (0.8, 402.857), (0.86, 482.143), (0.92, 561.429), (0.94, 640.714), (1.0, 720.0)],
    "TCOT": [(0.0, 97.0), (0.38, 175.286), (0.6, 253.571), (0.84, 331.857), (0.94, 410.143), (0.98, 488.429), (0.981, 566.714), (1.0, 645.0)],
    "TCAR": [(0.0, 104.0), (0.02, 153.0), (0.22, 202.0), (0.52, 251.0), (0.74, 300.0), (0.9, 349.0), (0.98, 398.0), (1.0, 447.0)],
    # Tiempos de descarga y pesaje en balanza (segundos)
    "TMIN": [(0.0, 13.0), (0.05, 28.0), (0.27, 43.0), (0.45, 58.0), (0.63, 73.0), (0.80, 88.0), (0.88, 103.0), (1.0, 118.0)],
    "TDESM": [(0.0, 20.0), (0.28, 57.8571), (0.76, 95.7143), (0.92, 133.571), (0.98, 171.429), (0.981, 209.286), (0.985, 247.143), (1.0, 285.0)],
    "TBALANZA": [(0.0, 76.0), (0.333, 96.8571), (0.5833, 117.714), (0.79167, 138.571), (0.8958, 159.429), (0.9792, 180.286), (0.9821, 201.143), (1.0, 222.0)],
    # Tonelaje de mineral (toneladas)
    "TN": [(0.0, 20.27), (0.12, 21.21), (0.2, 22.15), (0.28, 23.09), (0.48, 24.03), (0.78, 24.97), (0.9, 25.91), (1.0, 26.85)],
}


# ==============================================================================
# 2. ESPECIFICACIÓN DE DISTRIBUCIONES ESTOCÁSTICAS
# ==============================================================================
@dataclass
class DistributionSpec:
    """
    Representa una distribución empírica (ojiva CDF) o paramétrica.
    Tipos soportados: EMPIRICAL, CONSTANT, TRIANGULAR, NORMAL, LOGNORMAL.
    """
    dist_type: str
    points: Optional[List[Tuple[float, float]]] = None
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    min_val: float = 0.0
    unit: str = "s"
    category: str = "General"

    def __post_init__(self) -> None:
        self.dist_type = self.dist_type.strip().upper()
        if self.dist_type == "EMPIRICAL":
            if not self.points or len(self.points) < 2:
                raise ValueError("Una distribución EMPIRICAL requiere al menos 2 puntos (prob, valor).")
            self.points = sorted([(float(p), float(v)) for p, v in self.points], key=lambda x: x[0])

    def expected_value(self) -> float:
        """Valor esperado analítico de la distribución (usado para estimar ETA y cuotas proyectadas)."""
        dtype = self.dist_type
        if dtype == "CONSTANT":
            return float(self.param1)
        if dtype == "EMPIRICAL" and self.points:
            pts = self.points
            mean = 0.0
            for i in range(len(pts) - 1):
                p0, v0 = pts[i]
                p1, v1 = pts[i + 1]
                mean += (p1 - p0) * 0.5 * (v0 + v1)
            return float(mean)
        if dtype == "TRIANGULAR":
            return max(self.min_val, (self.param1 + self.param2 + self.param3) / 3.0)
        if dtype in ("NORMAL", "LOGNORMAL"):
            return max(self.min_val, float(self.param1))
        return max(self.min_val, float(self.param1))

    def sample(self, rng: random.Random) -> float:
        """Muestrea un valor usando el generador rng entregado."""
        dtype = self.dist_type

        if dtype == "CONSTANT":
            # No consume rng.random() para preservar paridad exacta con constantes del caso base
            return float(self.param1)

        if dtype == "EMPIRICAL":
            pts = self.points
            u = rng.random()
            for i in range(len(pts) - 1):
                p0, v0 = pts[i]
                p1, v1 = pts[i + 1]
                if p0 <= u <= p1:
                    if p1 == p0:
                        return v0
                    return v0 + ((u - p0) / (p1 - p0)) * (v1 - v0)
            return pts[-1][1]

        if dtype == "TRIANGULAR":
            low, mode, high = self.param1, self.param2, self.param3
            if high <= low:
                return max(self.min_val, low)
            mode = min(max(mode, low), high)
            return max(self.min_val, rng.triangular(low, high, mode))

        if dtype == "NORMAL":
            mean, std = self.param1, max(0.0, self.param2)
            if std == 0.0:
                return max(self.min_val, mean)
            return max(self.min_val, rng.gauss(mean, std))

        if dtype == "LOGNORMAL":
            mean, std = self.param1, max(0.0, self.param2)
            if mean <= 0.0:
                return max(self.min_val, 0.0)
            if std == 0.0:
                return max(self.min_val, mean)
            sigma_sq = math.log(1.0 + (std ** 2) / (mean ** 2))
            mu_ln = math.log(mean) - 0.5 * sigma_sq
            return max(self.min_val, rng.lognormvariate(mu_ln, math.sqrt(sigma_sq)))

        raise ValueError(f"Tipo de distribución no soportado: {dtype}")


# ==============================================================================
# 3. DATACLASSES DE CONFIGURACIÓN ABIERTA (N ZONAS + CONTINGENCIAS + ABET)
# ==============================================================================
@dataclass
class ZoneConfig:
    zone_id: str
    zone_name: str
    n_trucks: int = 1
    n_scoops: int = 1
    active: bool = True
    prob_mineral: float = 0.75
    quota_mineral_tn: float = 0.0
    max_quota_mineral_tn: float = 0.0  # 0.0 = sin tope superior
    ore_price_usd_tn: float = 60.0
    waste_tonnage_dist: Union[str, float] = 20.0
    ore_tonnage_dist: Optional[str] = None  # Si es None, usa global_cfg.dist_tonelaje_mineral
    entry_start_sec: float = 900.0
    entry_interval_sec: float = 240.0
    # 9 parámetros de ruta anclados en la bocamina/acceso de la zona
    # (soportan claves separadas por '|' y constantes numéricas en segundos, incluido 0)
    dist_inicial: List[str] = field(default_factory=list)                 # Estacionamiento -> Bocamina (superficie)
    dist_bajada_vacio: List[str] = field(default_factory=list)            # Bocamina -> Cámara de carguío (interior mina)
    dist_carguio: str = "0"                                               # Carguío en cámara (interior mina)
    dist_subida_cargado: List[str] = field(default_factory=list)          # Cámara de carguío -> Bocamina (interior mina)
    dist_hacia_balanza: List[str] = field(default_factory=list)           # Bocamina -> Balanza (superficie, mineral)
    dist_balanza_a_cancha: List[str] = field(default_factory=list)        # Balanza -> Cancha mineral (superficie, 0 sin balanza)
    dist_hacia_desmontera: List[str] = field(default_factory=list)        # Bocamina -> Desmontera (superficie, desmonte)
    dist_retorno_desde_mineral: List[str] = field(default_factory=list)   # Cancha mineral -> Bocamina (superficie, vacío)
    dist_retorno_desde_desmonte: List[str] = field(default_factory=list)  # Desmontera -> Bocamina (superficie, vacío)
    # Nombres de estación/cola (opcionales, autogenerados si están vacíos)
    facility_name: str = ""
    queue_name: str = ""
    # Parámetros ABET por zona/circuito
    hp_scoop: Optional[float] = None                 # Override opcional de potencia HP del scooptram de esta zona
    cap_max_ventilacion_m3_min: float = 0.0          # Tope de caudal de ventilación del circuito/zona (0 = sin tope)

    def __post_init__(self) -> None:
        self.zone_id = self.zone_id.strip().upper()
        self.dist_inicial = _normalize_route_chain(self.dist_inicial)
        self.dist_bajada_vacio = _normalize_route_chain(self.dist_bajada_vacio)
        if self.dist_carguio is None:
            self.dist_carguio = "0"
        elif isinstance(self.dist_carguio, (int, float)):
            self.dist_carguio = "0" if (isinstance(self.dist_carguio, float) and math.isnan(self.dist_carguio)) else str(self.dist_carguio)
        else:
            self.dist_carguio = str(self.dist_carguio).strip() or "0"
        self.dist_subida_cargado = _normalize_route_chain(self.dist_subida_cargado)
        self.dist_hacia_balanza = _normalize_route_chain(self.dist_hacia_balanza)
        self.dist_balanza_a_cancha = _normalize_route_chain(self.dist_balanza_a_cancha)
        self.dist_hacia_desmontera = _normalize_route_chain(self.dist_hacia_desmontera)
        self.dist_retorno_desde_mineral = _normalize_route_chain(self.dist_retorno_desde_mineral)
        self.dist_retorno_desde_desmonte = _normalize_route_chain(self.dist_retorno_desde_desmonte)
        self.cap_max_ventilacion_m3_min = max(0.0, float(self.cap_max_ventilacion_m3_min or 0.0))
        if not self.facility_name:
            self.facility_name = f"SCOOP_{self.zone_id}"
        if not self.queue_name:
            self.queue_name = f"Q_{self.zone_id}"


@dataclass
class BreakdownConfig:
    """
    Configuración de falla de equipo (scooptram o balanza).
    - mode='SCHEDULED': falla puntual What-If en start_min por duration_min.
    - mode='STOCHASTIC_MTBF': fallas exponenciales recurrentes según mtbf_hr y mttr_min.
    """
    event_id: str
    station_key: str  # zone_id (ej. 'HER') o 'BAL'
    mode: str = "SCHEDULED"
    start_min: float = 180.0
    duration_min: float = 45.0
    mtbf_hr: float = 0.0
    mttr_min: float = 0.0
    units_affected: int = 1
    active: bool = True

    def __post_init__(self) -> None:
        self.station_key = self.station_key.strip().upper()
        self.mode = self.mode.strip().upper()


@dataclass
class GlobalConfig:
    # Operación
    sim_time_sec: float = 32400.0
    meta_mineral_gdia: float = 1500.0
    guardias_anio: int = 730
    capacidad_volquete_tn: float = 25.0
    politica_despacho: str = "DIN_ETA_RITMO"    # FIJA, DIN_ETA_RITMO, DIN_ETA_ESPERA, DIN_TIEMPO_TOTAL, DIN_REACTIVA
    modo_parada: str = "HARD"                   # HARD (corte exacto) o SOFT (termina ciclo)
    avoid_down_zones: bool = True               # Internalizado siempre en True (desvío obligatorio ante fallas)
    semilla: int = 25
    num_replicas: int = 30
    nivel_confianza: float = 0.95

    # Infraestructura superficie
    num_balanzas: int = 1                       # 0 desactiva el pesaje en balanza
    cap_descarga_mineral: int = 0               # 0 = sin restricción de cola
    cap_descarga_desmonte: int = 0              # 0 = sin restricción de cola
    dist_balanza: str = "TBALANZA"
    dist_descarga_mineral: str = "TMIN"
    dist_descarga_desmonte: str = "TDESM"
    dist_tonelaje_mineral: str = "TN"

    # Economía unificada por estado del equipo (Opción A: HM Productivas vs. Standby/Cola)
    costo_op_volquete_hr: float = 50.0          # US$/HM — solo horas productivas de volquete
    costo_cola_volquete_hr: float = 30.0        # US$/h  — solo horas en cola/standby de volquete
    costo_op_scoop_hr: float = 160.0            # US$/HM — solo horas cargando de scooptram
    costo_improd_scoop_hr: float = 90.0         # US$/h  — solo horas en standby esperando volquete
    usar_tope_planta: bool = False              # True = tope exacto de capacidad de planta remunerada
    cap_planta_tn: float = 0.0                  # <= 0 usa meta_mineral_gdia como respaldo

    # Rango Sim-Opt (usado por optimizador.py)
    flota_min: int = 10
    flota_max: int = 18
    funcion_objetivo: str = "INGRESO_NETO"      # INGRESO_NETO, TONELAJE, COSTO_UNITARIO

    # Parámetros ABET: Ventilación (D.S. 023-2017-EM) y Emisiones (IPCC/MINAM)
    hp_volquete: float = 400.0
    dm_volquete: float = 0.933                  # Disponibilidad mecánica de flota (93.33%); NO descuenta caudal pico
    hp_scoop: float = 250.0
    dm_scoop: float = 0.85                      # Disponibilidad mecánica de scooptram; NO descuenta caudal pico
    factor_vent_m3_min_hp: float = 3.0          # D.S. 023-2017-EM (m3/min por HP)
    diesel_ralenti_l_hr: float = 3.5            # Consumo ralentí en cola de volquetes (L/h)
    precio_diesel_usd_l: float = 1.20           # Precio de diésel (US$/L)
    factor_co2_kg_l: float = 2.68               # Factor de emisión IPCC / MINAM (kg CO2/L)

    def __post_init__(self) -> None:
        self.politica_despacho = normalize_policy(self.politica_despacho)
        # Garantía incondicional: las políticas dinámicas siempre evitan zonas completamente caídas
        self.avoid_down_zones = True

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GlobalConfig":
        """Construye GlobalConfig ignorando limpiamente claves obsoletas de versiones previas."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class SimulationConfig:
    global_cfg: GlobalConfig
    zones: List[ZoneConfig]
    distributions: Dict[str, DistributionSpec]
    breakdowns: List[BreakdownConfig] = field(default_factory=list)

    def clone(self) -> "SimulationConfig":
        return copy.deepcopy(self)

    def set_fleet_assignment(self, assignment: Dict[str, int]) -> "SimulationConfig":
        """Retorna una copia con nueva asignación de volquetes por zone_id."""
        cfg = self.clone()
        for z in cfg.zones:
            if z.zone_id in assignment:
                z.n_trucks = int(assignment[z.zone_id])
        return cfg


# ==============================================================================
# 4. MONITOR DE INSTALACIONES Y COLAS (MULTI-SERVIDOR + PRIORITYRESOURCE)
# ==============================================================================
class StationMonitor:
    """
    Rastrea métricas de utilización, fallas y colas para capacidad k >= 1
    con simpy.PriorityResource.
    """
    PRIORITY_BREAKDOWN = 0
    PRIORITY_TRUCK = 10

    def __init__(self, env: simpy.Environment, fac_name: str, queue_name: str, capacity: int = 1):
        self.env = env
        self.fac_name = fac_name
        self.queue_name = queue_name
        self.capacity = max(1, int(capacity))
        self.resource = simpy.PriorityResource(env, capacity=self.capacity)

        # Métricas de Facility (servidores en trabajo productivo)
        self.fac_entries: int = 0
        self.active_busy_servers: int = 0
        self.busy_area_time: float = 0.0
        self.last_busy_change: float = 0.0
        self.current_owner: int = 0

        # Métricas de Fallas (Downtime por PriorityResource)
        self.active_down_servers: int = 0
        self.downtime_area_time: float = 0.0
        self.last_down_change: float = 0.0
        self.breakdown_count: int = 0

        # Métricas de Cola (Queue)
        self.q_entries: int = 0
        self.q_zero_entries: int = 0
        self.q_max: int = 0
        self.q_current: int = 0
        self.q_area_time: float = 0.0
        self.q_last_change: float = 0.0
        self.wait_times: List[float] = []

    def _update_queue_area(self, now: Optional[float] = None) -> None:
        t = self.env.now if now is None else now
        self.q_area_time += self.q_current * (t - self.q_last_change)
        self.q_last_change = t

    def _update_busy_area(self, now: Optional[float] = None) -> None:
        t = self.env.now if now is None else now
        self.busy_area_time += self.active_busy_servers * (t - self.last_busy_change)
        self.last_busy_change = t

    def _update_down_area(self, now: Optional[float] = None) -> None:
        t = self.env.now if now is None else now
        self.downtime_area_time += self.active_down_servers * (t - self.last_down_change)
        self.last_down_change = t

    def enter_queue(self) -> float:
        self._update_queue_area()
        self.q_entries += 1
        self.q_current += 1
        if self.q_current > self.q_max:
            self.q_max = self.q_current
        return self.env.now

    def depart_queue_and_seize(self, truck_id: int, arrival_time: float) -> float:
        self._update_queue_area()
        self.q_current -= 1
        wait = self.env.now - arrival_time
        self.wait_times.append(wait)
        if wait < 1e-7:
            self.q_zero_entries += 1

        self._update_busy_area()
        self.active_busy_servers += 1
        self.fac_entries += 1
        self.current_owner = truck_id
        return wait

    def release_facility(self) -> None:
        self._update_busy_area()
        self.active_busy_servers = max(0, self.active_busy_servers - 1)
        if self.active_busy_servers == 0:
            self.current_owner = 0

    def start_breakdown_unit(self) -> None:
        self._update_down_area()
        self.active_down_servers += 1
        self.breakdown_count += 1

    def end_breakdown_unit(self) -> None:
        self._update_down_area()
        self.active_down_servers = max(0, self.active_down_servers - 1)

    def is_fully_down(self) -> bool:
        return self.active_down_servers >= self.capacity

    def effective_available_capacity(self) -> int:
        return max(0, self.capacity - self.active_down_servers)

    def finalize_at_horizon(self, sim_horizon: float) -> None:
        now = min(self.env.now, sim_horizon) if sim_horizon > 0 else self.env.now
        self._update_queue_area(now)
        self._update_busy_area(now)
        self._update_down_area(now)


# ==============================================================================
# 5. MOTOR DES GENÉRICO PARA N ZONAS (SubHaulDESEngine)
# ==============================================================================
class SubHaulDESEngine:
    """
    Motor de simulación estocástica de eventos discretos para carguío y acarreo
    subterráneo configurable para N zonas, múltiples scooptrams, balanzas y fallas.
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed_override: Optional[int] = None,
        record_trace: bool = False,
        **_legacy_kwargs: object,
    ):
        # _legacy_kwargs ignora limpiamente parámetros obsoletos (ej. costo_op_volquete_hora)
        self.config = config
        self.g = config.global_cfg
        self.g.avoid_down_zones = True
        self.policy = normalize_policy(self.g.politica_despacho)
        self.g.politica_despacho = self.policy

        self.seed = self.g.semilla if seed_override is None else int(seed_override)
        self.rng = random.Random(self.seed)
        # RNG separado para fallas estocásticas MTBF/MTTR (no altera secuencia base de ciclos)
        self.rng_fail = random.Random(self.seed + 100_003)

        self.record_trace: bool = bool(record_trace)
        self.trace: Dict[str, List[Dict[str, object]]] = {
            "travels": [],
            "stations": [],
            "dumps": [],
            "breakdowns": [],
            "redirections": [],
        }

        self.env = simpy.Environment()
        self.sim_time = float(self.g.sim_time_sec)
        self.effective_horizon: float = self.sim_time

        self.active_zones: List[ZoneConfig] = [z for z in config.zones if z.active]
        if not self.active_zones:
            raise ValueError("Debe existir al menos una zona activa para ejecutar la simulación.")
        self.zone_map: Dict[str, ZoneConfig] = {z.zone_id: z for z in self.active_zones}

        # Sincronización automática de la meta diaria con la suma de cuotas activas por zona
        active_quota_sum = sum(float(z.quota_mineral_tn) for z in self.active_zones)
        if active_quota_sum > 0.0:
            self.g.meta_mineral_gdia = active_quota_sum

        # Estaciones: 1 por zona activa + balanza (si num_balanzas > 0) + descargas (si cap > 0)
        self.stations: Dict[str, StationMonitor] = {}
        for z in self.active_zones:
            self.stations[z.zone_id] = StationMonitor(
                self.env, fac_name=z.facility_name, queue_name=z.queue_name, capacity=z.n_scoops
            )

        if self.g.num_balanzas > 0:
            self.stations["BAL"] = StationMonitor(
                self.env, fac_name="BALANZA", queue_name="Q_BAL", capacity=self.g.num_balanzas
            )
        if self.g.cap_descarga_mineral > 0:
            self.stations["DUMP_ORE"] = StationMonitor(
                self.env, fac_name="CANCHA_MIN", queue_name="Q_DUMP_ORE", capacity=self.g.cap_descarga_mineral
            )
        if self.g.cap_descarga_desmonte > 0:
            self.stations["DUMP_WASTE"] = StationMonitor(
                self.env, fac_name="DESMONTERA", queue_name="Q_DUMP_WASTE", capacity=self.g.cap_descarga_desmonte
            )

        # Rastreo continuo de volquetes efectivamente dentro de interior mina por zona
        # (únicamente: dist_bajada_vacio + cola en cámara + dist_carguio + dist_subida_cargado)
        self.underground_trucks_current: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.underground_trucks_max: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.underground_trucks_area_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.underground_trucks_last_change: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}

        # Acumuladores de producción y métricas de ciclo
        self.ore_by_zone: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.waste_by_zone: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.trips_ore_by_zone: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.trips_waste_by_zone: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.time_to_target_sec: Optional[float] = None

        # Partición exclusiva de tiempo en volquetes: productivo (HM) vs. cola/standby
        self.zone_truck_wait_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.zone_truck_prod_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.scale_wait_sec: float = 0.0
        self.total_truck_productive_sec: float = 0.0
        self.total_truck_wait_sec: float = 0.0
        self.n_trucks_total: int = sum(z.n_trucks for z in self.active_zones)

        # Tope configurable de capacidad de planta remunerada (corte exacto en viaje de cruce)
        self.effective_plant_cap_tn: float = (
            float(self.g.cap_planta_tn) if float(self.g.cap_planta_tn) > 0 else float(self.g.meta_mineral_gdia)
        )
        self.capped_ore_tn: float = 0.0
        self.capped_gross_revenue_usd: float = 0.0

        # Acumuladores diagnósticos por zona (cola + servicio en balanza y descargas)
        self.zone_scale_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.zone_scale_cnt: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.zone_dump_ore_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.zone_dump_ore_cnt: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self.zone_dump_waste_sec: Dict[str, float] = {z.zone_id: 0.0 for z in self.active_zones}
        self.zone_dump_waste_cnt: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}

        # Estado predictivo en tiempo real (ETA + camiones asignados + cuota proyectada)
        self.in_flight_etas: Dict[str, List[Tuple[float, int]]] = {z.zone_id: [] for z in self.active_zones}
        self.active_assigned_trucks: Dict[str, int] = {z.zone_id: 0 for z in self.active_zones}
        self._precompute_expected_times()

    # --------------------------------------------------------------------------
    # Monitoreo continuo de ocupación en interior mina por zona (D.S. 023-2017-EM)
    # --------------------------------------------------------------------------
    def _update_underground_area(self, zid: str, now: Optional[float] = None) -> None:
        t = self.env.now if now is None else now
        self.underground_trucks_area_sec[zid] += (
            self.underground_trucks_current[zid] * (t - self.underground_trucks_last_change[zid])
        )
        self.underground_trucks_last_change[zid] = t

    def _enter_underground(self, zid: str) -> None:
        self._update_underground_area(zid)
        self.underground_trucks_current[zid] += 1
        if self.underground_trucks_current[zid] > self.underground_trucks_max[zid]:
            self.underground_trucks_max[zid] = self.underground_trucks_current[zid]

    def _exit_underground(self, zid: str) -> None:
        self._update_underground_area(zid)
        self.underground_trucks_current[zid] = max(0, self.underground_trucks_current[zid] - 1)

    # --------------------------------------------------------------------------
    # Valores esperados analíticos (para ETA y cuota proyectada en despacho)
    # --------------------------------------------------------------------------
    def expected_ref(self, ref: Union[str, float, int]) -> float:
        """Retorna el valor esperado de una referencia de distribución, cadena '|' o constante."""
        if isinstance(ref, (int, float)):
            return max(0.0, float(ref))
        key = str(ref).strip()
        if not key:
            return 0.0
        if "|" in key:
            return sum(self.expected_ref(part) for part in key.split("|") if part.strip())
        if key in self.config.distributions:
            return max(0.0, self.config.distributions[key].expected_value())
        try:
            return max(0.0, float(key))
        except ValueError:
            return 0.0

    def _expected_chain(self, segments: Sequence[str]) -> float:
        return sum(self.expected_ref(seg) for seg in segments if str(seg).strip())

    def _precompute_expected_times(self) -> None:
        self.exp_init: Dict[str, float] = {}
        self.exp_bajada: Dict[str, float] = {}
        self.exp_load: Dict[str, float] = {}
        self.exp_ret_desde_min: Dict[str, float] = {}
        self.exp_ret_desde_desm: Dict[str, float] = {}
        self.exp_ore_trip: Dict[str, float] = {}

        for z in self.active_zones:
            zid = z.zone_id
            self.exp_init[zid] = self._expected_chain(z.dist_inicial)
            self.exp_bajada[zid] = self._expected_chain(z.dist_bajada_vacio)
            self.exp_load[zid] = max(0.0, self.expected_ref(z.dist_carguio))
            self.exp_ret_desde_min[zid] = self._expected_chain(z.dist_retorno_desde_mineral)
            self.exp_ret_desde_desm[zid] = self._expected_chain(z.dist_retorno_desde_desmonte)
            ore_ref = z.ore_tonnage_dist or self.g.dist_tonelaje_mineral
            self.exp_ore_trip[zid] = z.prob_mineral * self.expected_ref(ore_ref)

    def _est_travel_from_dump(self, is_mineral: bool, target_zone_id: str) -> float:
        """
        Tiempo esperado de viaje vacío desde el punto de descarga (cancha o desmontera)
        hasta la cámara destino `target_zone_id`, vinculado exclusivamente a la zona destino:
        - Desde mineral : E[dist_retorno_desde_mineral[j]]  + E[dist_bajada_vacio[j]]
        - Desde desmonte: E[dist_retorno_desde_desmonte[j]] + E[dist_bajada_vacio[j]]
        """
        t_retorno = (
            self.exp_ret_desde_min[target_zone_id]
            if is_mineral
            else self.exp_ret_desde_desm[target_zone_id]
        )
        return t_retorno + self.exp_bajada[target_zone_id]

    def _est_queue_clear_time(self, zid: str, eta_truck: float) -> float:
        """
        Estima el instante T_libre en que un scooptram de la zona `zid` quedará
        desocupado para atender a un volquete cuyo arribo estimado es `eta_truck`,
        considerando camiones cargando, en cola y en tránsito con ETA anterior.
        """
        st = self.stations[zid]
        cap = max(1, st.effective_available_capacity())
        t_load = self.exp_load[zid]
        now = self.env.now

        busy_cnt = min(cap, st.active_busy_servers)
        servers_free = [now + 0.5 * t_load] * busy_cnt + [now] * (cap - busy_cnt)
        servers_free.sort()

        for _ in range(st.q_current):
            t0 = servers_free.pop(0)
            servers_free.append(t0 + t_load)
            servers_free.sort()

        for arr_t, _ in sorted(self.in_flight_etas[zid], key=lambda item: item[0]):
            if arr_t <= eta_truck:
                t0 = servers_free.pop(0)
                start = max(t0, arr_t)
                servers_free.append(start + t_load)
                servers_free.sort()

        return servers_free[0]

    def _projected_ore(self, z: ZoneConfig) -> float:
        """Mineral descargado más el aporte esperado de volquetes actualmente asignados a la zona."""
        return self.ore_by_zone[z.zone_id] + self.active_assigned_trucks[z.zone_id] * self.exp_ore_trip[z.zone_id]

    # --------------------------------------------------------------------------
    # Muestreo de distribuciones y cadenas de tramos
    # --------------------------------------------------------------------------
    def sample_ref(self, ref: Union[str, float, int]) -> float:
        """Muestrea una referencia que puede ser una clave, una cadena '|', o un número constante."""
        if isinstance(ref, (int, float)):
            return max(0.0, float(ref))
        key = str(ref).strip()
        if not key:
            return 0.0
        if "|" in key:
            return sum(self.sample_ref(part) for part in key.split("|") if part.strip())
        if key in self.config.distributions:
            return max(0.0, self.config.distributions[key].sample(self.rng))
        try:
            return max(0.0, float(key))
        except ValueError as exc:
            raise KeyError(f"Distribución '{key}' no encontrada en SimulationConfig.distributions.") from exc

    def travel_segments(
        self,
        segments: Sequence[str],
        zone_for_prod: Optional[str] = None,
        truck_id: Optional[int] = None,
        role: Optional[str] = None,
        zone_id: Optional[str] = None,
        load_state: str = "EMPTY",
        redirected: bool = False,
        prev_zone_id: Optional[str] = None,
    ):
        """Ejecuta secuencialmente una lista de tramos de viaje y registra tiempo productivo."""
        for raw_seg in segments:
            for seg in str(raw_seg).split("|"):
                seg = seg.strip()
                if not seg:
                    continue
                dt = self.sample_ref(seg)
                if dt > 0.0:
                    t0 = self.env.now
                    if self.record_trace and role is not None and truck_id is not None:
                        self.trace["travels"].append({
                            "truck_id": int(truck_id),
                            "t0": round(t0, 2),
                            "t1": round(t0 + dt, 2),
                            "seg_key": seg,
                            "role": role,
                            "zone_id": zone_id if zone_id is not None else zone_for_prod,
                            "load_state": load_state,
                            "redirected": bool(redirected),
                            "prev_zone_id": prev_zone_id,
                        })
                    yield self.env.timeout(dt)
                    elapsed = self.env.now - t0
                    self.total_truck_productive_sec += elapsed
                    if zone_for_prod is not None and zone_for_prod in self.zone_truck_prod_sec:
                        self.zone_truck_prod_sec[zone_for_prod] += elapsed

    def use_station(
        self,
        station_key: str,
        truck_id: int,
        service_ref: Union[str, float],
        zone_for_wait: Optional[str] = None,
        load_state: str = "EMPTY",
    ):
        """
        Secuencia QUEUE -> SEIZE (PriorityResource) -> DEPART -> ADVANCE -> RELEASE.
        Partición estricta: `wait_time` va exclusivamente a cola; `elapsed` va
        exclusivamente a tiempo productivo (HM).
        """
        st = self.stations[station_key]
        t_arr = st.enter_queue()

        trace_ev: Optional[Dict[str, object]] = None
        if self.record_trace:
            trace_ev = {
                "truck_id": int(truck_id),
                "station_key": station_key,
                "t_enter_q": round(t_arr, 2),
                "t_start_svc": None,
                "t_end_svc": None,
                "zone_id": zone_for_wait if zone_for_wait is not None else station_key,
                "load_state": load_state,
            }
            self.trace["stations"].append(trace_ev)

        req = st.resource.request(priority=StationMonitor.PRIORITY_TRUCK)
        yield req

        wait_time = st.depart_queue_and_seize(truck_id, t_arr)
        self.total_truck_wait_sec += wait_time
        if zone_for_wait is not None and zone_for_wait in self.zone_truck_wait_sec:
            self.zone_truck_wait_sec[zone_for_wait] += wait_time
        if station_key == "BAL":
            self.scale_wait_sec += wait_time

        service_time = self.sample_ref(service_ref)
        t0 = self.env.now
        if trace_ev is not None:
            trace_ev["t_start_svc"] = round(t0, 2)
            trace_ev["t_end_svc"] = round(t0 + service_time, 2)

        if service_time > 0.0:
            yield self.env.timeout(service_time)
        elapsed = self.env.now - t0
        self.total_truck_productive_sec += elapsed
        if zone_for_wait is not None and zone_for_wait in self.zone_truck_prod_sec:
            self.zone_truck_prod_sec[zone_for_wait] += elapsed

        st.release_facility()
        st.resource.release(req)
        return elapsed

    # --------------------------------------------------------------------------
    # Procesos de contingencia / falla de equipos (simpy.PriorityResource)
    # --------------------------------------------------------------------------
    def _breakdown_unit_process(self, station_key: str, start_sec: float, duration_sec: float):
        """Bloquea 1 unidad de la estación con prioridad máxima durante duration_sec."""
        if station_key not in self.stations or duration_sec <= 0:
            return
        yield self.env.timeout(max(0.0, start_sec))
        st = self.stations[station_key]
        req = st.resource.request(priority=StationMonitor.PRIORITY_BREAKDOWN)
        yield req
        st.start_breakdown_unit()
        if self.record_trace:
            t_start = self.env.now
            self.trace["breakdowns"].append({
                "station_key": station_key,
                "t_start": round(t_start, 2),
                "t_end": round(t_start + duration_sec, 2),
            })
        yield self.env.timeout(duration_sec)
        st.end_breakdown_unit()
        st.resource.release(req)

    def _stochastic_mtbf_process(self, bd: BreakdownConfig):
        """Genera fallas recurrentes exponenciales (MTBF/MTTR) durante la guardia."""
        if bd.station_key not in self.stations or bd.mtbf_hr <= 0 or bd.mttr_min <= 0:
            return
        st = self.stations[bd.station_key]
        mtbf_sec = bd.mtbf_hr * 3600.0
        mttr_sec = bd.mttr_min * 60.0

        while self.env.now < self.sim_time:
            ttf = self.rng_fail.expovariate(1.0 / mtbf_sec)
            yield self.env.timeout(ttf)
            if self.env.now >= self.sim_time:
                break
            ttr = self.rng_fail.expovariate(1.0 / mttr_sec)
            req = st.resource.request(priority=StationMonitor.PRIORITY_BREAKDOWN)
            yield req
            st.start_breakdown_unit()
            if self.record_trace:
                t_start = self.env.now
                self.trace["breakdowns"].append({
                    "station_key": bd.station_key,
                    "t_start": round(t_start, 2),
                    "t_end": round(t_start + ttr, 2),
                })
            yield self.env.timeout(ttr)
            st.end_breakdown_unit()
            st.resource.release(req)

    # --------------------------------------------------------------------------
    # Políticas de despacho (5 estrategias: FIJA, DIN_REACTIVA + 3 predictivas)
    # --------------------------------------------------------------------------
    def choose_next_zone_reactive(self, current_zone: Optional[str] = None) -> str:
        """
        Decisión reactiva evaluada tras descargar (en cancha o desmontera):
        - FIJA: retorna siempre current_zone (B = A).
        - DIN_REACTIVA: evita incondicionalmente zonas caídas (con fallback solo si
          todas caen) y elige zona B usando colas instantáneas y cuotas descargadas.
        """
        if self.policy == "FIJA":
            if current_zone is None:
                raise ValueError("La política FIJA requiere current_zone.")
            return current_zone

        candidates = list(self.active_zones)

        # Filtro 1 incondicional de resiliencia: evitar zonas completamente detenidas
        operational = [z for z in candidates if not self.stations[z.zone_id].is_fully_down()]
        if operational:
            candidates = operational

        under_max = [
            z for z in candidates
            if z.max_quota_mineral_tn <= 0.0 or self.ore_by_zone[z.zone_id] < z.max_quota_mineral_tn
        ]
        if under_max:
            candidates = under_max
        unmet = [
            z for z in candidates
            if z.quota_mineral_tn > 0.0 and self.ore_by_zone[z.zone_id] < z.quota_mineral_tn
        ]
        if unmet:
            candidates = unmet

        def queue_score(idx_zone: Tuple[int, ZoneConfig]) -> Tuple[float, int, float, int]:
            idx, z = idx_zone
            st = self.stations[z.zone_id]
            avail_cap = max(1, st.effective_available_capacity())
            q_frac = (self.ore_by_zone[z.zone_id] / z.quota_mineral_tn) if z.quota_mineral_tn > 0.0 else 1.0
            return (st.q_current / avail_cap, st.q_current, q_frac, idx)

        _, best_zone = min(enumerate(candidates), key=queue_score)
        return best_zone.zone_id

    def choose_next_zone_predictive(self, is_mineral: bool) -> str:
        """
        Decisión predictiva ejecutada inmediatamente después de descargar (en cancha o desmontera),
        totalmente desvinculada de la zona de origen A.
        Todas las variantes evitan incondicionalmente zonas caídas y usan cuota proyectada:
        - DIN_ETA_RITMO    : Ritmo horario de cuota -> menor espera al llegar (ETA) -> menor viaje.
        - DIN_ETA_ESPERA   : Menor espera al llegar (ETA) -> menor % cuota proyectada -> menor viaje.
        - DIN_TIEMPO_TOTAL : Menor tiempo total estimado (t_viaje + t_cola proyectado).
        """
        candidates = list(self.active_zones)

        # 1. Filtro 1 incondicional de resiliencia: evitar zonas completamente detenidas por falla
        operational = [z for z in candidates if not self.stations[z.zone_id].is_fully_down()]
        if operational:
            candidates = operational

        # 2. Respetar tope máximo de extracción si está configurado (> 0)
        under_max = [
            z for z in candidates
            if z.max_quota_mineral_tn <= 0.0 or self._projected_ore(z) < z.max_quota_mineral_tn
        ]
        if under_max:
            candidates = under_max

        # 3. Bloqueo temporal de zonas que ya cumplieron su cuota proyectada de guardia
        #    (se rehabilitan automáticamente cuando todas las zonas completan su cuota)
        unmet = [
            z for z in candidates
            if z.quota_mineral_tn > 0.0 and self._projected_ore(z) < z.quota_mineral_tn
        ]
        if unmet:
            candidates = unmet

        now = self.env.now

        def score(idx_zone: Tuple[int, ZoneConfig]) -> Tuple[float, ...]:
            idx, z = idx_zone
            zid = z.zone_id
            st = self.stations[zid]
            cap = max(1, st.effective_available_capacity())
            t_travel = self._est_travel_from_dump(is_mineral, zid)
            eta = now + t_travel
            t_load = self.exp_load[zid]
            proj_ore = self._projected_ore(z)

            if self.policy == "DIN_TIEMPO_TOTAL":
                n_in_flight = len(self.in_flight_etas[zid])
                proj_wait = ((st.q_current + 0.5 * st.active_busy_servers + n_in_flight) / cap) * t_load
                return (t_travel + proj_wait, float(idx))

            t_free = self._est_queue_clear_time(zid, eta)
            exp_wait = max(0.0, t_free - eta)
            wait_bin = round(exp_wait / 15.0)

            if self.policy == "DIN_ETA_ESPERA":
                q_frac = (proj_ore / z.quota_mineral_tn) if z.quota_mineral_tn > 0.0 else 1.0
                return (float(wait_bin), q_frac, t_travel, float(idx))

            # DIN_ETA_RITMO (estilo Modular DISPATCH)
            pace_target = z.quota_mineral_tn * min(1.0, (now + 1800.0) / max(1.0, self.sim_time))
            behind = 0.0 if (z.quota_mineral_tn > 0.0 and proj_ore < pace_target) else 1.0
            return (behind, float(wait_bin), t_travel, float(idx))

        _, best_zone = min(enumerate(candidates), key=score)
        return best_zone.zone_id

    # --------------------------------------------------------------------------
    # Ciclo operativo del volquete (N zonas, anclado en bocamina)
    # --------------------------------------------------------------------------
    def truck_process(self, truck_id: int, initial_zone_id: str, entry_delay: float):
        yield self.env.timeout(entry_delay)
        current_zone_id = initial_zone_id
        prev_zone_for_bajada: Optional[str] = None
        redirected_for_bajada: bool = False
        soft_stop = (self.g.modo_parada.strip().upper() == "SOFT")
        is_predictive = (self.policy in PREDICTIVE_POLICIES)

        # Registro inicial de asignación y ETA para políticas predictivas
        my_eta_entry: Optional[Tuple[float, int]] = None
        if is_predictive:
            init_eta = self.env.now + self.exp_init[current_zone_id] + self.exp_bajada[current_zone_id]
            my_eta_entry = (init_eta, truck_id)
            self.in_flight_etas[current_zone_id].append(my_eta_entry)
            self.active_assigned_trucks[current_zone_id] += 1

        # 1. Traslado inicial (superficie): Estacionamiento -> Bocamina[A]
        yield from self.travel_segments(
            self.zone_map[current_zone_id].dist_inicial,
            zone_for_prod=current_zone_id,
            truck_id=truck_id,
            role="inicial",
            zone_id=current_zone_id,
            load_state="EMPTY",
            redirected=False,
            prev_zone_id=None,
        )

        # 2. Bucle de ciclos operativos durante la guardia
        while self.env.now < self.sim_time:
            z = self.zone_map[current_zone_id]

            # --- INGRESO A INTERIOR MINA (Bocamina[A] -> Cámara[A] -> Bocamina[A]) ---
            self._enter_underground(z.zone_id)

            # Bocamina[A] -> Cámara de carguío[A] (vacío, interior mina)
            yield from self.travel_segments(
                z.dist_bajada_vacio,
                zone_for_prod=z.zone_id,
                truck_id=truck_id,
                role="bajada",
                zone_id=z.zone_id,
                load_state="EMPTY",
                redirected=redirected_for_bajada,
                prev_zone_id=prev_zone_for_bajada,
            )

            # El volquete arribó a la cámara: sale de la lista en tránsito antes de entrar a cola
            if is_predictive and my_eta_entry is not None:
                if my_eta_entry in self.in_flight_etas[z.zone_id]:
                    self.in_flight_etas[z.zone_id].remove(my_eta_entry)
                my_eta_entry = None

            # Cola y Carguío en cámara[A] (interior mina)
            yield from self.use_station(
                z.zone_id, truck_id, z.dist_carguio, zone_for_wait=z.zone_id, load_state="EMPTY"
            )

            # Decisión probabilística mineral vs. desmonte
            is_mineral = (self.rng.random() <= z.prob_mineral)
            mat_state = "ORE" if is_mineral else "WASTE"

            # Cámara de carguío[A] -> Bocamina[A] (cargado, interior mina)
            yield from self.travel_segments(
                z.dist_subida_cargado,
                zone_for_prod=z.zone_id,
                truck_id=truck_id,
                role="subida",
                zone_id=z.zone_id,
                load_state=mat_state,
                redirected=False,
                prev_zone_id=z.zone_id,
            )

            # --- SALIDA DE INTERIOR MINA HACIA SUPERFICIE (Bocamina[A]) ---
            self._exit_underground(z.zone_id)

            if is_mineral:
                # Bocamina[A] -> Balanza -> Cancha de mineral -> Descarga mineral (superficie)
                yield from self.travel_segments(
                    z.dist_hacia_balanza,
                    zone_for_prod=z.zone_id,
                    truck_id=truck_id,
                    role="hacia_balanza",
                    zone_id=z.zone_id,
                    load_state="ORE",
                    redirected=False,
                    prev_zone_id=z.zone_id,
                )
                if self.g.num_balanzas > 0:
                    t_ini = self.env.now
                    yield from self.use_station(
                        "BAL", truck_id, self.g.dist_balanza, zone_for_wait=z.zone_id, load_state="ORE"
                    )
                    self.zone_scale_sec[z.zone_id] += self.env.now - t_ini
                    self.zone_scale_cnt[z.zone_id] += 1
                yield from self.travel_segments(
                    z.dist_balanza_a_cancha,
                    zone_for_prod=z.zone_id,
                    truck_id=truck_id,
                    role="balanza_a_cancha",
                    zone_id=z.zone_id,
                    load_state="ORE",
                    redirected=False,
                    prev_zone_id=z.zone_id,
                )

                t_ini = self.env.now
                if self.g.cap_descarga_mineral > 0:
                    dump_svc_sec = yield from self.use_station(
                        "DUMP_ORE",
                        truck_id,
                        self.g.dist_descarga_mineral,
                        zone_for_wait=z.zone_id,
                        load_state="ORE",
                    )
                else:
                    yield from self.travel_segments([self.g.dist_descarga_mineral], zone_for_prod=z.zone_id)
                    dump_svc_sec = self.env.now - t_ini
                self.zone_dump_ore_sec[z.zone_id] += self.env.now - t_ini
                self.zone_dump_ore_cnt[z.zone_id] += 1

                ore_dist = z.ore_tonnage_dist or self.g.dist_tonelaje_mineral
                ton = self.sample_ref(ore_dist)

                # Corte exacto en el tope de planta con NSR de la zona de origen del viaje
                if self.effective_plant_cap_tn > 0.0:
                    rem_cap = max(0.0, self.effective_plant_cap_tn - self.capped_ore_tn)
                    if rem_cap > 0.0:
                        paid_ton = min(ton, rem_cap)
                        self.capped_ore_tn += paid_ton
                        self.capped_gross_revenue_usd += paid_ton * z.ore_price_usd_tn

                # Producción física total acumulada en toda la guardia
                self.ore_by_zone[z.zone_id] += ton
                if (
                    self.time_to_target_sec is None
                    and self.g.meta_mineral_gdia > 0.0
                    and sum(self.ore_by_zone.values()) >= self.g.meta_mineral_gdia
                ):
                    self.time_to_target_sec = float(self.env.now)
                self.trips_ore_by_zone[z.zone_id] += 1
                if self.record_trace:
                    self.trace["dumps"].append({
                        "t": round(self.env.now, 2),
                        "truck_id": int(truck_id),
                        "zone_id": z.zone_id,
                        "material": "ORE",
                        "tons": round(ton, 3),
                        "dump_duration_sec": round(dump_svc_sec, 2),
                    })
            else:
                # Bocamina[A] -> Desmontera -> Descarga desmonte (superficie)
                yield from self.travel_segments(
                    z.dist_hacia_desmontera,
                    zone_for_prod=z.zone_id,
                    truck_id=truck_id,
                    role="hacia_desmontera",
                    zone_id=z.zone_id,
                    load_state="WASTE",
                    redirected=False,
                    prev_zone_id=z.zone_id,
                )

                t_ini = self.env.now
                if self.g.cap_descarga_desmonte > 0:
                    dump_svc_sec = yield from self.use_station(
                        "DUMP_WASTE",
                        truck_id,
                        self.g.dist_descarga_desmonte,
                        zone_for_wait=z.zone_id,
                        load_state="WASTE",
                    )
                else:
                    yield from self.travel_segments([self.g.dist_descarga_desmonte], zone_for_prod=z.zone_id)
                    dump_svc_sec = self.env.now - t_ini
                self.zone_dump_waste_sec[z.zone_id] += self.env.now - t_ini
                self.zone_dump_waste_cnt[z.zone_id] += 1

                ton_w = self.sample_ref(z.waste_tonnage_dist)
                self.waste_by_zone[z.zone_id] += ton_w
                self.trips_waste_by_zone[z.zone_id] += 1
                if self.record_trace:
                    self.trace["dumps"].append({
                        "t": round(self.env.now, 2),
                        "truck_id": int(truck_id),
                        "zone_id": z.zone_id,
                        "material": "WASTE",
                        "tons": round(ton_w, 3),
                        "dump_duration_sec": round(dump_svc_sec, 2),
                    })

            # --- DESVINCULACIÓN Y DECISIÓN TRAS DESCARGAR ---
            if is_predictive:
                self.active_assigned_trucks[z.zone_id] = max(0, self.active_assigned_trucks[z.zone_id] - 1)

            if soft_stop and self.env.now >= self.sim_time:
                break

            if self.policy == "FIJA":
                next_zone_id = current_zone_id
            elif is_predictive:
                next_zone_id = self.choose_next_zone_predictive(is_mineral)
                est_t = self._est_travel_from_dump(is_mineral, next_zone_id)
                my_eta_entry = (self.env.now + est_t, truck_id)
                self.in_flight_etas[next_zone_id].append(my_eta_entry)
                self.active_assigned_trucks[next_zone_id] += 1
            else:
                next_zone_id = self.choose_next_zone_reactive()

            was_redirected = (next_zone_id != current_zone_id)
            if self.record_trace and was_redirected:
                self.trace["redirections"].append({
                    "t": round(self.env.now, 2),
                    "truck_id": int(truck_id),
                    "from_zone": current_zone_id,
                    "to_zone": next_zone_id,
                    "dump_point": "CANCHA_MIN" if is_mineral else "DESMONTERA",
                })

            # --- RETORNO EN SUPERFICIE (vinculado EXCLUSIVAMENTE a la zona destino B) ---
            next_z = self.zone_map[next_zone_id]
            if is_mineral:
                yield from self.travel_segments(
                    next_z.dist_retorno_desde_mineral,
                    zone_for_prod=next_zone_id,
                    truck_id=truck_id,
                    role="retorno_min",
                    zone_id=next_zone_id,
                    load_state="EMPTY",
                    redirected=was_redirected,
                    prev_zone_id=current_zone_id,
                )
            else:
                yield from self.travel_segments(
                    next_z.dist_retorno_desde_desmonte,
                    zone_for_prod=next_zone_id,
                    truck_id=truck_id,
                    role="retorno_desm",
                    zone_id=next_zone_id,
                    load_state="EMPTY",
                    redirected=was_redirected,
                    prev_zone_id=current_zone_id,
                )

            prev_zone_for_bajada = current_zone_id
            redirected_for_bajada = was_redirected
            current_zone_id = next_zone_id

    # --------------------------------------------------------------------------
    # Ejecución principal
    # --------------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        """Lanza procesos de volquetes y contingencias, ejecuta SimPy y cierra integradores."""
        truck_id = 1
        for z in self.active_zones:
            for i in range(int(z.n_trucks)):
                delay = z.entry_start_sec + i * z.entry_interval_sec
                self.env.process(self.truck_process(truck_id, z.zone_id, delay))
                truck_id += 1

        # Registrar contingencias activas (fallas de scooptram / balanza)
        for bd in self.config.breakdowns:
            if not bd.active or bd.station_key not in self.stations:
                continue
            units = max(1, min(int(bd.units_affected), self.stations[bd.station_key].capacity))
            if bd.mode == "SCHEDULED":
                for _ in range(units):
                    self.env.process(
                        self._breakdown_unit_process(
                            bd.station_key, bd.start_min * 60.0, bd.duration_min * 60.0
                        )
                    )
            elif bd.mode == "STOCHASTIC_MTBF":
                for _ in range(units):
                    self.env.process(self._stochastic_mtbf_process(bd))

        if self.g.modo_parada.strip().upper() == "SOFT":
            self.env.run()
            self.effective_horizon = max(self.sim_time, float(self.env.now))
        else:
            self.env.run(until=self.sim_time)
            self.effective_horizon = self.sim_time

        for st in self.stations.values():
            st.finalize_at_horizon(self.effective_horizon)

        for z in self.active_zones:
            self._update_underground_area(z.zone_id, self.effective_horizon)

        return self.get_summary_kpis()

    # --------------------------------------------------------------------------
    # Payload de eventos (trace) serializable a JSON para visualización 2D
    # --------------------------------------------------------------------------
    def get_trace_payload(self) -> Dict[str, object]:
        """
        Retorna un diccionario ligero y 100% serializable con `json.dumps(...)`
        conteniendo metadatos del escenario, KPIs de verificación y listas de eventos.
        """
        kpis = self.get_summary_kpis()
        return {
            "policy": str(self.policy),
            "policy_label": str(POLICY_LABELS.get(self.policy, self.policy)),
            "stop_mode": str(self.g.modo_parada),
            "sim_time_sec": float(self.sim_time),
            "effective_horizon_sec": float(self.effective_horizon),
            "seed": int(self.seed),
            "n_trucks_total": int(self.n_trucks_total),
            "num_balanzas": int(self.g.num_balanzas),
            "zones": [
                {
                    "id": str(z.zone_id),
                    "name": str(z.zone_name),
                    "n_trucks": int(z.n_trucks),
                    "n_scoops": int(z.n_scoops),
                    "quota_mineral_tn": float(z.quota_mineral_tn),
                    "exp_bajada_sec": round(float(self.exp_bajada.get(z.zone_id, 0.0)), 2),
                }
                for z in self.active_zones
            ],
            "kpi_summary": {
                "ore_total_tn": float(kpis["ore_total_tn"]),
                "capped_ore_tn": float(kpis["capped_ore_tn"]),
                "waste_total_tn": float(kpis["waste_total_tn"]),
                "total_material_tn": float(kpis["total_material_tn"]),
                "ore_by_zone": {str(k): float(v) for k, v in kpis["ore_by_zone"].items()},
                "waste_by_zone": {str(k): float(v) for k, v in kpis["waste_by_zone"].items()},
                "target_met": bool(kpis["target_met"]),
                "gross_revenue_usd": float(kpis["gross_revenue_usd"]),
                "total_opex_usd": float(kpis["total_opex_usd"]),
                "net_revenue_usd": float(kpis["net_revenue_usd"]),
                "unit_cost_usd_tn": float(kpis["unit_cost_usd_tn"]),
                "total_truck_wait_hr": float(kpis["total_truck_wait_hr"]),
                "truck_fu": float(kpis["truck_fu"]),
                "vent_q_peak_total_m3_min": float(kpis["vent_q_peak_total_m3_min"]),
                "vent_q_peak_total_cfm": float(kpis["vent_q_peak_total_cfm"]),
            },
            "travels": self.trace["travels"],
            "stations": self.trace["stations"],
            "dumps": self.trace["dumps"],
            "breakdowns": self.trace["breakdowns"],
            "redirections": self.trace["redirections"],
        }

    # --------------------------------------------------------------------------
    # Reportes y KPIs estructurados (Estaciones + Económicos + ABET)
    # --------------------------------------------------------------------------
    def _ordered_station_keys(self) -> List[str]:
        keys = [z.zone_id for z in self.active_zones]
        for extra in ("BAL", "DUMP_ORE", "DUMP_WASTE"):
            if extra in self.stations:
                keys.append(extra)
        return keys

    def get_facilities_dataframe(self) -> pd.DataFrame:
        """Reporte de utilización de instalaciones (compatibilidad interna)."""
        rows = []
        for key in self._ordered_station_keys():
            st = self.stations[key]
            denom = st.capacity * self.effective_horizon
            util = (st.busy_area_time / denom) if denom > 0 else 0.0
            down_pct = (st.downtime_area_time / denom) * 100.0 if denom > 0 else 0.0
            ave_time = (st.busy_area_time / st.fac_entries) if st.fac_entries > 0 else 0.0
            rows.append({
                "FACILITY": st.fac_name,
                "CAP.": st.capacity,
                "ENTRIES": st.fac_entries,
                "UTIL.": round(util, 3),
                "AVE. TIME": round(ave_time, 3),
                "DOWNTIME (%)": round(down_pct, 2),
                "AVAIL.": st.effective_available_capacity(),
                "OWNER": st.current_owner,
                "DELAY": st.q_current,
            })
        return pd.DataFrame(rows)

    def get_queues_dataframe(self) -> pd.DataFrame:
        """Reporte de colas (compatibilidad interna)."""
        rows = []
        for key in self._ordered_station_keys():
            st = self.stations[key]
            ave_cont = st.q_area_time / self.effective_horizon if self.effective_horizon > 0 else 0.0
            ave_time = float(np.mean(st.wait_times)) if st.wait_times else 0.0
            non_zero = [w for w in st.wait_times if w >= 1e-7]
            ave_non_zero = float(np.mean(non_zero)) if non_zero else 0.0
            rows.append({
                "QUEUE": st.queue_name,
                "MAX": st.q_max,
                "CONT.": st.q_current,
                "ENTRY": st.q_entries,
                "ENTRY (0)": st.q_zero_entries,
                "AVE.CONT.": round(ave_cont, 3),
                "AVE.TIME": round(ave_time, 3),
                "AVE.(-0)": round(ave_non_zero, 3),
            })
        return pd.DataFrame(rows)

    def get_stations_report_dataframe(self) -> pd.DataFrame:
        """Reporte unificado de estaciones, colas y demoras de superficie por zona."""
        surface_labels = {
            "BAL": "Balanza",
            "DUMP_ORE": "Cancha Mineral",
            "DUMP_WASTE": "Desmontera",
        }
        rows = []
        for key in self._ordered_station_keys():
            st = self.stations[key]
            is_zone = key in self.zone_map

            denom = st.capacity * self.effective_horizon
            util = (st.busy_area_time / denom) if denom > 0 else 0.0
            down_pct = (st.downtime_area_time / denom) * 100.0 if denom > 0 else 0.0
            ave_service = (st.busy_area_time / st.fac_entries) if st.fac_entries > 0 else 0.0

            zero_wait_pct = (st.q_zero_entries / st.q_entries * 100.0) if st.q_entries > 0 else 100.0
            cola_max = 0 if st.q_entries == st.q_zero_entries else st.q_max
            ave_wait_all = float(np.mean(st.wait_times)) if st.wait_times else 0.0
            non_zero = [w for w in st.wait_times if w >= 1e-7]
            ave_wait_nonzero = float(np.mean(non_zero)) if non_zero else 0.0

            if is_zone:
                estacion_lbl = f"Scooptram ({self.zone_map[key].zone_name})"
                sc_cnt = self.zone_scale_cnt[key]
                ore_cnt = self.zone_dump_ore_cnt[key]
                wst_cnt = self.zone_dump_waste_cnt[key]
                tot_dump_cnt = ore_cnt + wst_cnt

                t_balanza: Optional[float] = (
                    round(self.zone_scale_sec[key] / sc_cnt, 1) if sc_cnt > 0 else 0.0
                )
                t_desc_min: Optional[float] = (
                    round(self.zone_dump_ore_sec[key] / ore_cnt, 1) if ore_cnt > 0 else 0.0
                )
                t_desc_desm: Optional[float] = (
                    round(self.zone_dump_waste_sec[key] / wst_cnt, 1) if wst_cnt > 0 else 0.0
                )
                t_desc_total: Optional[float] = (
                    round((self.zone_dump_ore_sec[key] + self.zone_dump_waste_sec[key]) / tot_dump_cnt, 1)
                    if tot_dump_cnt > 0
                    else 0.0
                )
            else:
                estacion_lbl = surface_labels.get(key, st.fac_name)
                t_balanza = None
                t_desc_min = None
                t_desc_desm = None
                t_desc_total = None

            rows.append({
                "Estación": estacion_lbl,
                "Equipos": st.capacity,
                "Atención": st.fac_entries,
                "Uso (%)": round(util * 100.0, 2),
                "Falla (%)": round(down_pct, 2),
                "T. Servicio (s)": round(ave_service, 1),
                "Sin Cola (%)": round(zero_wait_pct, 1),
                "Cola Máx.": cola_max,
                "Espera Prom. (s)": round(ave_wait_all, 1),
                "Espera Real (s)": round(ave_wait_nonzero, 1),
                "T. Balanza (s)": t_balanza,
                "T. Desc. Min. (s)": t_desc_min,
                "T. Desc. Desm. (s)": t_desc_desm,
                "T. Desc. Total (s)": t_desc_total,
            })
        return pd.DataFrame(rows)

    def get_savevalues_dataframe(self) -> pd.DataFrame:
        """Reporte de tonelajes acumulados por zona y totales."""
        rows = []
        for z in self.active_zones:
            rows.append({"SAVEVALUE": f"ORE_{z.zone_id}", "VALUE": round(self.ore_by_zone[z.zone_id], 3)})
            rows.append({"SAVEVALUE": f"WASTE_{z.zone_id}", "VALUE": round(self.waste_by_zone[z.zone_id], 3)})
        rows.append({"SAVEVALUE": "ORE_TOTAL", "VALUE": round(sum(self.ore_by_zone.values()), 3)})
        rows.append({"SAVEVALUE": "DESMONTE_TOTAL", "VALUE": round(sum(self.waste_by_zone.values()), 3)})
        return pd.DataFrame(rows)

    def get_economic_dataframe(self, include_extended_costs: bool = False) -> pd.DataFrame:
        """
        Evaluación productiva y económica por zona bajo el esquema unificado
        por estado del equipo (Opción A: HM productivas vs. Standby/Cola).
        """
        rows = []

        tot_trucks = 0
        tot_ore = 0.0
        tot_waste = 0.0
        tot_rev = 0.0
        tot_scoop_op = 0.0
        tot_scoop_idle = 0.0
        tot_truck_op = 0.0
        tot_truck_wait = 0.0
        tot_opex = 0.0

        for z in self.active_zones:
            st = self.stations[z.zone_id]
            cap_horizon_sec = st.capacity * self.effective_horizon
            util_pct = (st.busy_area_time / cap_horizon_sec * 100.0) if cap_horizon_sec > 0 else 0.0

            ore_tn = self.ore_by_zone[z.zone_id]
            waste_tn = self.waste_by_zone[z.zone_id]
            rev = ore_tn * z.ore_price_usd_tn

            # Scooptrams: busy_sec + idle_sec + down_sec == capacity * effective_horizon
            busy_sec = st.busy_area_time
            down_sec = st.downtime_area_time
            idle_sec = max(0.0, cap_horizon_sec - busy_sec - down_sec)
            c_scoop_op = (busy_sec / 3600.0) * self.g.costo_op_scoop_hr
            c_scoop_idle = (idle_sec / 3600.0) * self.g.costo_improd_scoop_hr

            # Volquetes: partición exclusiva productivo (HM) vs. cola/standby
            c_truck_op = (self.zone_truck_prod_sec[z.zone_id] / 3600.0) * self.g.costo_op_volquete_hr
            c_truck_wait = (self.zone_truck_wait_sec[z.zone_id] / 3600.0) * self.g.costo_cola_volquete_hr

            c_total = c_scoop_op + c_scoop_idle + c_truck_op + c_truck_wait
            net = rev - c_total

            tot_trucks += z.n_trucks
            tot_ore += ore_tn
            tot_waste += waste_tn
            tot_rev += rev
            tot_scoop_op += c_scoop_op
            tot_scoop_idle += c_scoop_idle
            tot_truck_op += c_truck_op
            tot_truck_wait += c_truck_wait
            tot_opex += c_total

            row: Dict[str, object] = {
                "ZONA": z.zone_name.upper(),
                "Volquetes asignados": z.n_trucks,
                "Utilización scooptram (%)": f"{util_pct:.2f}%",
                "Producción mineral (tn)": round(ore_tn, 2),
                "Ingreso producción (US$)": round(rev, 2),
                "Costo Op. Scoop (US$)": round(c_scoop_op, 2),
                "Costo Standby Scoop (US$)": round(c_scoop_idle, 2),
                "Costo Op. Volq. (US$)": round(c_truck_op, 2),
                "Costo Cola Volq. (US$)": round(c_truck_wait, 2),
                "Costo Total (US$)": round(c_total, 2),
                "Ingreso Neto (US$)": round(net, 2),
            }
            if include_extended_costs:
                row["Producción desmonte (tn)"] = round(waste_tn, 2)
            rows.append(row)

        tot_gross_eff = self.capped_gross_revenue_usd if self.g.usar_tope_planta else tot_rev
        tot_net_eff = tot_gross_eff - tot_opex

        total_row: Dict[str, object] = {
            "ZONA": "TOTAL",
            "Volquetes asignados": tot_trucks,
            "Utilización scooptram (%)": "-",
            "Producción mineral (tn)": round(tot_ore, 2),
            "Ingreso producción (US$)": round(tot_gross_eff, 2),
            "Costo Op. Scoop (US$)": round(tot_scoop_op, 2),
            "Costo Standby Scoop (US$)": round(tot_scoop_idle, 2),
            "Costo Op. Volq. (US$)": round(tot_truck_op, 2),
            "Costo Cola Volq. (US$)": round(tot_truck_wait, 2),
            "Costo Total (US$)": round(tot_opex, 2),
            "Ingreso Neto (US$)": round(tot_net_eff, 2),
        }
        if include_extended_costs:
            total_row["Producción desmonte (tn)"] = round(tot_waste, 2)
        rows.append(total_row)

        return pd.DataFrame(rows)

    def get_economic_table(self, include_extended_costs: bool = False) -> pd.DataFrame:
        """Alias compatible con llamadas previas hacia la tabla económica."""
        return self.get_economic_dataframe(include_extended_costs=include_extended_costs)

    def get_summary_kpis(self) -> Dict[str, object]:
        """Diccionario consolidado de KPIs productivos, económicos, ventilación D.S. 023 y CO2."""
        T_real_horas = self.effective_horizon / 3600.0
        ore_total = sum(self.ore_by_zone.values())
        waste_total = sum(self.waste_by_zone.values())

        uncapped_gross_revenue_usd = sum(
            self.ore_by_zone[z.zone_id] * z.ore_price_usd_tn for z in self.active_zones
        )

        # 1. Scooptrams: busy_sec + idle_sec + down_sec == capacity * effective_horizon
        scoop_op_cost = 0.0
        scoop_idle_cost = 0.0
        scoop_util_by_zone: Dict[str, float] = {}
        scoop_downtime_hr_by_zone: Dict[str, float] = {}

        for z in self.active_zones:
            st = self.stations[z.zone_id]
            cap_horizon_sec = st.capacity * self.effective_horizon
            busy_sec = st.busy_area_time
            down_sec = st.downtime_area_time
            idle_sec = max(0.0, cap_horizon_sec - busy_sec - down_sec)

            scoop_util_by_zone[z.zone_id] = (busy_sec / cap_horizon_sec) if cap_horizon_sec > 0 else 0.0
            scoop_downtime_hr_by_zone[z.zone_id] = down_sec / 3600.0
            scoop_op_cost += (busy_sec / 3600.0) * self.g.costo_op_scoop_hr
            scoop_idle_cost += (idle_sec / 3600.0) * self.g.costo_improd_scoop_hr

        # 2. Volquetes: partición exclusiva productivo (HM) vs. cola/standby
        truck_prod_sec = sum(self.zone_truck_prod_sec.values())
        truck_wait_sec = sum(self.zone_truck_wait_sec.values())
        truck_wait_hr = truck_wait_sec / 3600.0
        truck_op_cost = (truck_prod_sec / 3600.0) * self.g.costo_op_volquete_hr
        truck_wait_cost = truck_wait_hr * self.g.costo_cola_volquete_hr
        fleet_op_cost = truck_op_cost + truck_wait_cost

        # 3. OPEX Total, Ingreso Neto y Costo Unitario
        total_opex = scoop_op_cost + scoop_idle_cost + truck_op_cost + truck_wait_cost
        uncapped_net_revenue_usd = uncapped_gross_revenue_usd - total_opex
        capped_gross = self.capped_gross_revenue_usd
        capped_net = capped_gross - total_opex

        if self.g.usar_tope_planta:
            gross_rev = capped_gross
            net_revenue = capped_net
        else:
            gross_rev = uncapped_gross_revenue_usd
            net_revenue = uncapped_net_revenue_usd

        unit_cost = (total_opex / ore_total) if ore_total > 0 else 0.0

        total_truck_shift_sec = max(1.0, self.n_trucks_total * self.effective_horizon)
        truck_fu = min(1.0, truck_prod_sec / total_truck_shift_sec)

        scale_util = 0.0
        if "BAL" in self.stations:
            st_bal = self.stations["BAL"]
            scale_util = st_bal.busy_area_time / (st_bal.capacity * self.effective_horizon)

        max_queue_by_station = {k: st.q_max for k, st in self.stations.items()}
        avg_wait_nonzero_sec = {}
        zero_wait_pct = {}
        for k, st in self.stations.items():
            nz = [w for w in st.wait_times if w >= 1e-7]
            avg_wait_nonzero_sec[k] = float(np.mean(nz)) if nz else 0.0
            zero_wait_pct[k] = (st.q_zero_entries / st.q_entries * 100.0) if st.q_entries > 0 else 100.0

        # 4. Ventilación normativa D.S. 023-2017-EM por pico simultáneo local en interior mina
        #    Q_pico,zona = factor_vent * (N_scoop,zona * HP_scoop,zona + N_volq_int_max,zona * HP_volq)
        #    (Fu = 1.0, sin promediar en la guardia y sin aplicar Dm como factor reductor)
        f_vent = float(self.g.factor_vent_m3_min_hp)
        hp_volq = float(self.g.hp_volquete)

        underground_trucks_peak_by_zone: Dict[str, int] = {}
        underground_trucks_avg_by_zone: Dict[str, float] = {}
        vent_q_peak_m3_min_by_zone: Dict[str, float] = {}
        vent_q_peak_cfm_by_zone: Dict[str, float] = {}
        vent_cap_max_m3_min_by_zone: Dict[str, float] = {}
        vent_cap_usage_pct_by_zone: Dict[str, float] = {}
        vent_margin_m3_min_by_zone: Dict[str, Optional[float]] = {}
        vent_alert_by_zone: Dict[str, bool] = {}

        for z in self.active_zones:
            zid = z.zone_id
            hp_sc = float(z.hp_scoop) if z.hp_scoop is not None else float(self.g.hp_scoop)
            peak_volq = int(self.underground_trucks_max[zid])
            avg_volq = (
                self.underground_trucks_area_sec[zid] / self.effective_horizon
                if self.effective_horizon > 0.0
                else 0.0
            )
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

        vent_q_peak_total_m3_min = sum(vent_q_peak_m3_min_by_zone.values())
        vent_q_peak_total_cfm = vent_q_peak_total_m3_min * M3_MIN_TO_CFM
        vent_any_alert = any(vent_alert_by_zone.values())

        # 5. Combustible desperdiciado y CO2 en colas (EXCLUSIVAMENTE horas en cola de volquetes;
        #    excluye completamente tiempo idle/standby de scooptrams)
        guardias_anio = int(self.g.guardias_anio)
        diesel_wasted_wait_l_shift = truck_wait_hr * float(self.g.diesel_ralenti_l_hr)
        diesel_wasted_wait_l_year = diesel_wasted_wait_l_shift * guardias_anio
        diesel_wasted_wait_cost_usd_shift = diesel_wasted_wait_l_shift * float(self.g.precio_diesel_usd_l)
        diesel_wasted_wait_cost_usd_year = diesel_wasted_wait_cost_usd_shift * guardias_anio
        co2_wasted_wait_kg_shift = diesel_wasted_wait_l_shift * float(self.g.factor_co2_kg_l)
        co2_wasted_wait_t_year = (co2_wasted_wait_kg_shift * guardias_anio) / 1000.0

        return {
            "seed": self.seed,
            "policy": self.policy,
            "policy_label": POLICY_LABELS.get(self.policy, self.policy),
            "stop_mode": self.g.modo_parada,
            "effective_horizon_sec": self.effective_horizon,
            "T_real_horas": round(T_real_horas, 4),
            "n_trucks_total": self.n_trucks_total,
            "dm_volquete": round(float(self.g.dm_volquete), 4),
            "assignment": {z.zone_id: z.n_trucks for z in self.active_zones},
            "ore_total_tn": round(ore_total, 4),
            "capped_ore_tn": round(self.capped_ore_tn, 4),
            "waste_total_tn": round(waste_total, 4),
            "total_material_tn": round(ore_total + waste_total, 4),
            "ore_by_zone": {k: round(v, 4) for k, v in self.ore_by_zone.items()},
            "waste_by_zone": {k: round(v, 4) for k, v in self.waste_by_zone.items()},
            "target_met": ore_total >= self.g.meta_mineral_gdia,
            "time_to_target_sec": round(self.time_to_target_sec, 2) if self.time_to_target_sec is not None else None,
            "time_to_target_hr": round(self.time_to_target_sec / 3600.0, 4) if self.time_to_target_sec is not None else None,
            "gross_revenue_usd": round(gross_rev, 2),
            "scoop_op_cost_usd": round(scoop_op_cost, 2),
            "scoop_idle_cost_usd": round(scoop_idle_cost, 2),
            "truck_op_cost_usd": round(truck_op_cost, 2),
            "truck_wait_cost_usd": round(truck_wait_cost, 2),
            "fleet_op_cost_usd": round(fleet_op_cost, 2),
            "total_opex_usd": round(total_opex, 2),
            "net_revenue_usd": round(net_revenue, 2),
            "capped_net_revenue_usd": round(capped_net, 2),
            "uncapped_net_revenue_usd": round(uncapped_net_revenue_usd, 2),
            "unit_cost_usd_tn": round(unit_cost, 4),
            "total_truck_prod_hr": round(truck_prod_sec / 3600.0, 4),
            "total_truck_wait_hr": round(truck_wait_hr, 4),
            "wait_hr_by_zone": {k: round(v / 3600.0, 4) for k, v in self.zone_truck_wait_sec.items()},
            "scale_wait_hr": round(self.scale_wait_sec / 3600.0, 4),
            "scoop_utilization_by_zone": {k: round(v, 4) for k, v in scoop_util_by_zone.items()},
            "scoop_downtime_hr_by_zone": {k: round(v, 4) for k, v in scoop_downtime_hr_by_zone.items()},
            "scale_utilization": round(scale_util, 4),
            "truck_fu": round(truck_fu, 4),
            "max_queue_by_station": max_queue_by_station,
            "avg_wait_nonzero_sec_by_station": {k: round(v, 2) for k, v in avg_wait_nonzero_sec.items()},
            "zero_wait_pct_by_station": {k: round(v, 2) for k, v in zero_wait_pct.items()},
            # Ventilación normativa D.S. 023-2017-EM (pico simultáneo interior mina por zona)
            "underground_trucks_peak_by_zone": underground_trucks_peak_by_zone,
            "underground_trucks_avg_by_zone": underground_trucks_avg_by_zone,
            "vent_q_peak_m3_min_by_zone": vent_q_peak_m3_min_by_zone,
            "vent_q_peak_cfm_by_zone": vent_q_peak_cfm_by_zone,
            "vent_q_peak_total_m3_min": round(vent_q_peak_total_m3_min, 2),
            "vent_q_peak_total_cfm": round(vent_q_peak_total_cfm, 1),
            "vent_cap_max_m3_min_by_zone": vent_cap_max_m3_min_by_zone,
            "vent_cap_usage_pct_by_zone": vent_cap_usage_pct_by_zone,
            "vent_margin_m3_min_by_zone": vent_margin_m3_min_by_zone,
            "vent_alert_by_zone": vent_alert_by_zone,
            "vent_any_alert": vent_any_alert,
            # Combustible desperdiciado y CO2 en colas de volquetes (guardia y año)
            "diesel_wasted_wait_l_shift": round(diesel_wasted_wait_l_shift, 2),
            "diesel_wasted_wait_l_year": round(diesel_wasted_wait_l_year, 1),
            "diesel_wasted_wait_cost_usd_shift": round(diesel_wasted_wait_cost_usd_shift, 2),
            "diesel_wasted_wait_cost_usd_year": round(diesel_wasted_wait_cost_usd_year, 2),
            "co2_wasted_wait_kg_shift": round(co2_wasted_wait_kg_shift, 2),
            "co2_wasted_wait_t_year": round(co2_wasted_wait_t_year, 3),
        }


# ==============================================================================
# 6. PRESET DEL CASO BASE VALIDADO (HUANCAPETÍ — ESQUEMA BOCAMINA DE 9 RUTAS)
# ==============================================================================
def build_huancapeti_base_config(
    n_her: int = 6,
    n_car: int = 2,
    n_cot: int = 7,
    policy: str = "DIN_ETA_RITMO",
    prob_mode: str = "FORECAST",
    seed: int = 25,
    stop_mode: str = "HARD",
) -> SimulationConfig:
    """
    Construye el caso base validado de U.M. Huancapetí sin hardcodear el motor.
    - prob_mode='GPSS_WORLD_VAL': probabilidades del escenario histórico 12 volquetes.
    - prob_mode='FORECAST': probabilidades de expansión 15 volquetes (Tablas 22–31).
    """
    if prob_mode.upper() == "GPSS_WORLD_VAL":
        probs = {"HER": 0.75, "CAR": 0.65, "COT": 0.83}
    else:
        probs = {"HER": 0.70, "CAR": 0.75, "COT": 0.86}

    distributions: Dict[str, DistributionSpec] = {}
    for name, pts in HUANCAPETI_CDF_FUNCTIONS.items():
        if name in ("THER", "TCOT", "TCAR"):
            cat, unit = "Carguío", "s"
        elif name in ("TMIN", "TDESM", "TBALANZA"):
            cat, unit = "Descarga/Balanza", "s"
        elif name == "TN":
            cat, unit = "Tonelaje", "t"
        else:
            cat, unit = "Traslado", "s"
        distributions[name] = DistributionSpec(
            dist_type="EMPIRICAL", points=pts, unit=unit, category=cat
        )

    zones = [
        ZoneConfig(
            zone_id="HER",
            zone_name="Hércules",
            n_trucks=n_her,
            n_scoops=1,
            prob_mineral=probs["HER"],
            quota_mineral_tn=400.0,
            ore_price_usd_tn=62.0,
            waste_tonnage_dist=20.25,
            entry_start_sec=900.0,
            entry_interval_sec=180.0,
            dist_inicial=["T1"],
            dist_bajada_vacio=["T2"],
            dist_carguio="THER",
            dist_subida_cargado=["T3"],
            dist_hacia_balanza=["T4", "T4HKC", "T9"],
            dist_balanza_a_cancha=["T10"],
            dist_hacia_desmontera=["T4", "T4HKC", "T5"],
            dist_retorno_desde_mineral=["T11", "T8HKC", "T8"],
            dist_retorno_desde_desmonte=["T6", "T8HKC", "T8"],
            facility_name="STSEISH",
            queue_name="Q_HER",
        ),
        ZoneConfig(
            zone_id="CAR",
            zone_name="Caridad",
            n_trucks=n_car,
            n_scoops=1,
            prob_mineral=probs["CAR"],
            quota_mineral_tn=300.0,
            ore_price_usd_tn=75.0,
            waste_tonnage_dist=19.38,
            entry_start_sec=900.0,
            entry_interval_sec=300.0,
            dist_inicial=["T1KC", "T4HKC", "T2K"],
            dist_bajada_vacio=["T3K"],
            dist_carguio="TCAR",
            dist_subida_cargado=["T4K"],
            dist_hacia_balanza=["T7K", "T9"],
            dist_balanza_a_cancha=["T10"],
            dist_hacia_desmontera=["T5K"],
            dist_retorno_desde_mineral=["T11", "T2K"],
            dist_retorno_desde_desmonte=["T6K"],
            facility_name="STSEISKAR",
            queue_name="Q_CAR",
        ),
        ZoneConfig(
            zone_id="COT",
            zone_name="Coturcán",
            n_trucks=n_cot,
            n_scoops=1,
            prob_mineral=probs["COT"],
            quota_mineral_tn=800.0,
            ore_price_usd_tn=60.0,
            waste_tonnage_dist=18.90,
            entry_start_sec=900.0,
            entry_interval_sec=240.0,
            dist_inicial=["T1KC"],
            dist_bajada_vacio=["T2C"],
            dist_carguio="TCOT",
            dist_subida_cargado=["T3C"],
            dist_hacia_balanza=["T4HKC", "T9"],
            dist_balanza_a_cancha=["T10"],
            dist_hacia_desmontera=["T4HKC", "T5"],
            dist_retorno_desde_mineral=["T11", "T8HKC"],
            dist_retorno_desde_desmonte=["T6", "T8HKC"],
            facility_name="STSEISCOT",
            queue_name="Q_COT",
        ),
    ]

    global_cfg = GlobalConfig(
        sim_time_sec=32400.0,
        meta_mineral_gdia=sum(float(z.quota_mineral_tn) for z in zones if z.active),
        guardias_anio=730,
        capacidad_volquete_tn=25.0,
        politica_despacho=normalize_policy(policy),
        modo_parada=stop_mode,
        semilla=seed,
        dm_volquete=0.933,
    )

    return SimulationConfig(global_cfg=global_cfg, zones=zones, distributions=distributions, breakdowns=[])


# ==============================================================================
# 7. EXPORTACIÓN E IMPORTACIÓN DE PLANTILLA EXCEL (.xlsx)
# ==============================================================================
def _as_bool(val: object, default: bool = False) -> bool:
    if pd.isna(val):
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().upper() in ("TRUE", "1", "SI", "SÍ", "YES", "Y")


def _as_float(val: object, default: float = 0.0) -> float:
    if pd.isna(val) or str(val).strip() == "":
        return default
    return float(val)


def _as_int(val: object, default: int = 0) -> int:
    if pd.isna(val) or str(val).strip() == "":
        return default
    return int(round(float(val)))


def _as_str(val: object, default: str = "") -> str:
    if pd.isna(val):
        return default
    return str(val).strip()


def _parse_chain(val: object) -> List[str]:
    if pd.isna(val):
        return []
    text = _as_str(val, "")
    if not text:
        return []
    return [p.strip() for p in text.split("|") if p.strip()]


def export_excel_template(
    filepath: Union[str, Path],
    config: Optional[SimulationConfig] = None,
) -> Path:
    """
    Genera la plantilla Excel de 5 hojas precargada con `config` (o con el
    caso base Huancapetí si `config` es None).
    """
    cfg = build_huancapeti_base_config() if config is None else config
    g = cfg.global_cfg
    out_path = Path(filepath)

    # Hoja 1: Parámetros Globales
    global_rows = [
        ("sim_time_sec", g.sim_time_sec, "s", "Duración efectiva de guardia (ej. 32400 = 9 h)"),
        ("meta_mineral_gdia", g.meta_mineral_gdia, "t/gdia", "Meta de extracción de mineral por guardia (suma de cuotas activas)"),
        ("guardias_anio", g.guardias_anio, "gdia/año", "Guardias operativas por año"),
        ("capacidad_volquete_tn", g.capacidad_volquete_tn, "t", "Capacidad nominal del volquete"),
        (
            "politica_despacho",
            normalize_policy(g.politica_despacho),
            "texto",
            "DIN_ETA_RITMO | DIN_ETA_ESPERA | DIN_TIEMPO_TOTAL | DIN_REACTIVA | FIJA",
        ),
        ("modo_parada", g.modo_parada, "texto", "HARD (corte exacto) | SOFT (culmina descarga en tránsito)"),
        ("semilla", g.semilla, "entero", "Semilla aleatoria Mersenne Twister"),
        ("num_replicas", g.num_replicas, "entero", "Réplicas de Montecarlo para IC y pruebas de hipótesis"),
        ("nivel_confianza", g.nivel_confianza, "frac", "Nivel de confianza estadística (ej. 0.95)"),
        ("num_balanzas", g.num_balanzas, "und", "Número de balanzas en superficie (0 desactiva pesaje)"),
        ("cap_descarga_mineral", g.cap_descarga_mineral, "und", "Bahías simultáneas cancha mineral (0 = sin cola)"),
        ("cap_descarga_desmonte", g.cap_descarga_desmonte, "und", "Bahías simultáneas desmontera (0 = sin cola)"),
        ("dist_balanza", g.dist_balanza, "ref", "Distribución de tiempo de pesaje en balanza"),
        ("dist_descarga_mineral", g.dist_descarga_mineral, "ref", "Distribución de descarga en cancha de mineral"),
        ("dist_descarga_desmonte", g.dist_descarga_desmonte, "ref", "Distribución de descarga en desmontera"),
        ("dist_tonelaje_mineral", g.dist_tonelaje_mineral, "ref", "Distribución global de tonelaje de mineral"),
        ("costo_op_volquete_hr", g.costo_op_volquete_hr, "US$/HM", "Tarifa operativa por hora productiva de volquete"),
        ("costo_cola_volquete_hr", g.costo_cola_volquete_hr, "US$/h", "Tarifa standby/cola por hora de volquete"),
        ("costo_op_scoop_hr", g.costo_op_scoop_hr, "US$/HM", "Tarifa operativa por hora cargando de scooptram"),
        ("costo_improd_scoop_hr", g.costo_improd_scoop_hr, "US$/h", "Tarifa standby por hora de scooptram esperando volquete"),
        ("usar_tope_planta", g.usar_tope_planta, "bool", "Aplicar tope exacto de capacidad de planta remunerada"),
        ("cap_planta_tn", g.cap_planta_tn, "t/gdia", "Capacidad de planta remunerada (0 = usa meta_mineral_gdia)"),
        ("flota_min", g.flota_min, "und", "Límite inferior de búsqueda de flota (Sim-Opt)"),
        ("flota_max", g.flota_max, "und", "Límite superior de búsqueda de flota (Sim-Opt)"),
        ("funcion_objetivo", g.funcion_objetivo, "texto", "INGRESO_NETO | TONELAJE | COSTO_UNITARIO"),
        ("hp_volquete", g.hp_volquete, "HP", "Potencia efectiva de volquete (D.S. 023-2017-EM)"),
        ("dm_volquete", g.dm_volquete, "frac", "Disponibilidad mecánica de volquete (solo flota; NO descuenta Q pico)"),
        ("hp_scoop", g.hp_scoop, "HP", "Potencia efectiva de scooptram (D.S. 023-2017-EM)"),
        ("dm_scoop", g.dm_scoop, "frac", "Disponibilidad mecánica de scooptram (NO descuenta Q pico)"),
        ("factor_vent_m3_min_hp", g.factor_vent_m3_min_hp, "m3/min/HP", "Requerimiento normativo (D.S. 023-2017-EM)"),
        ("diesel_ralenti_l_hr", g.diesel_ralenti_l_hr, "L/h", "Consumo de diésel en ralentí (solo colas de volquetes)"),
        ("precio_diesel_usd_l", g.precio_diesel_usd_l, "US$/L", "Precio del diésel"),
        ("factor_co2_kg_l", g.factor_co2_kg_l, "kgCO2/L", "Factor de emisión IPCC / MINAM"),
    ]
    df_global = pd.DataFrame(global_rows, columns=["parametro", "valor", "unidad", "descripcion"])

    # Hoja 2: Zonas (9 parámetros de ruta anclados en bocamina + tope de ventilación por circuito)
    zone_rows = []
    for z in cfg.zones:
        zone_rows.append({
            "zone_id": z.zone_id,
            "zone_name": z.zone_name,
            "active": z.active,
            "n_trucks": z.n_trucks,
            "n_scoops": z.n_scoops,
            "prob_mineral": z.prob_mineral,
            "quota_mineral_tn": z.quota_mineral_tn,
            "max_quota_mineral_tn": z.max_quota_mineral_tn,
            "ore_price_usd_tn": z.ore_price_usd_tn,
            "waste_tonnage_dist": z.waste_tonnage_dist,
            "ore_tonnage_dist": z.ore_tonnage_dist or "",
            "entry_start_sec": z.entry_start_sec,
            "entry_interval_sec": z.entry_interval_sec,
            "dist_inicial": "|".join(z.dist_inicial),
            "dist_bajada_vacio": "|".join(z.dist_bajada_vacio),
            "dist_carguio": z.dist_carguio,
            "dist_subida_cargado": "|".join(z.dist_subida_cargado),
            "dist_hacia_balanza": "|".join(z.dist_hacia_balanza),
            "dist_balanza_a_cancha": "|".join(z.dist_balanza_a_cancha),
            "dist_hacia_desmontera": "|".join(z.dist_hacia_desmontera),
            "dist_retorno_desde_mineral": "|".join(z.dist_retorno_desde_mineral),
            "dist_retorno_desde_desmonte": "|".join(z.dist_retorno_desde_desmonte),
            "facility_name": z.facility_name,
            "queue_name": z.queue_name,
            "hp_scoop": "" if z.hp_scoop is None else z.hp_scoop,
            "cap_max_ventilacion_m3_min": z.cap_max_ventilacion_m3_min,
        })
    df_zones = pd.DataFrame(zone_rows)

    # Hoja 3: Distribuciones Paramétricas (incluye ejemplos editables si no hay)
    param_rows = []
    for name, spec in cfg.distributions.items():
        if spec.dist_type != "EMPIRICAL":
            param_rows.append({
                "dist_name": name,
                "dist_type": spec.dist_type,
                "param1": spec.param1,
                "param2": spec.param2,
                "param3": spec.param3,
                "min_val": spec.min_val,
                "unidad": spec.unit,
                "notas": spec.category,
            })
    if not param_rows:
        param_rows = [
            {"dist_name": "EJ_TRIANGULAR", "dist_type": "TRIANGULAR", "param1": 180.0, "param2": 260.0, "param3": 420.0, "min_val": 60.0, "unidad": "s", "notas": "Ejemplo (min, moda, max)"},
            {"dist_name": "EJ_NORMAL", "dist_type": "NORMAL", "param1": 900.0, "param2": 120.0, "param3": 0.0, "min_val": 300.0, "unidad": "s", "notas": "Ejemplo (media, desv)"},
            {"dist_name": "EJ_LOGNORMAL", "dist_type": "LOGNORMAL", "param1": 1400.0, "param2": 180.0, "param3": 0.0, "min_val": 500.0, "unidad": "s", "notas": "Ejemplo (media, desv física)"},
        ]
    df_param = pd.DataFrame(param_rows)

    # Hoja 4: Ojivas Empíricas (formato largo)
    emp_rows = []
    for name, spec in cfg.distributions.items():
        if spec.dist_type == "EMPIRICAL" and spec.points:
            for prob, val in spec.points:
                emp_rows.append({
                    "dist_name": name,
                    "prob_acum": prob,
                    "valor": val,
                    "unidad": spec.unit,
                    "categoria": spec.category,
                })
    df_emp = pd.DataFrame(emp_rows, columns=["dist_name", "prob_acum", "valor", "unidad", "categoria"])

    # Hoja 5: Contingencias / Fallas
    bd_rows = []
    for bd in cfg.breakdowns:
        bd_rows.append({
            "event_id": bd.event_id,
            "station_key": bd.station_key,
            "mode": bd.mode,
            "start_min": bd.start_min,
            "duration_min": bd.duration_min,
            "mtbf_hr": bd.mtbf_hr,
            "mttr_min": bd.mttr_min,
            "units_affected": bd.units_affected,
            "active": bd.active,
        })
    if not bd_rows:
        bd_rows = [{
            "event_id": "EJ_FALLA_HER",
            "station_key": "HER",
            "mode": "SCHEDULED",
            "start_min": 180.0,
            "duration_min": 45.0,
            "mtbf_hr": 0.0,
            "mttr_min": 0.0,
            "units_affected": 1,
            "active": False,
        }]
    df_bd = pd.DataFrame(bd_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_global.to_excel(writer, sheet_name="1_Parametros_Globales", index=False)
        df_zones.to_excel(writer, sheet_name="2_Zonas", index=False)
        df_param.to_excel(writer, sheet_name="3_Distribuciones_Parametricas", index=False)
        df_emp.to_excel(writer, sheet_name="4_Ojivas_Empiricas", index=False)
        df_bd.to_excel(writer, sheet_name="5_Contingencias_Fallas", index=False)

        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 3, 45)

    return out_path


def load_config_from_excel(filepath: Union[str, Path]) -> SimulationConfig:
    """
    Lee una plantilla Excel de SubHaul-Opt y construye un SimulationConfig completo.
    Ignora limpiamente parámetros obsoletos de versiones previas y sincroniza
    `meta_mineral_gdia` con la suma de cuotas de zonas activas si esta es > 0.
    """
    xls = pd.ExcelFile(filepath, engine="openpyxl")

    # 1. Parámetros Globales
    df_g = pd.read_excel(xls, sheet_name="1_Parametros_Globales")
    g_map = {str(r["parametro"]).strip(): r["valor"] for _, r in df_g.iterrows() if pd.notna(r["parametro"])}

    global_cfg = GlobalConfig(
        sim_time_sec=_as_float(g_map.get("sim_time_sec"), 32400.0),
        meta_mineral_gdia=_as_float(g_map.get("meta_mineral_gdia"), 1500.0),
        guardias_anio=_as_int(g_map.get("guardias_anio"), 730),
        capacidad_volquete_tn=_as_float(g_map.get("capacidad_volquete_tn"), 25.0),
        politica_despacho=normalize_policy(_as_str(g_map.get("politica_despacho"), "DIN_ETA_RITMO")),
        modo_parada=_as_str(g_map.get("modo_parada"), "HARD").upper(),
        avoid_down_zones=True,
        semilla=_as_int(g_map.get("semilla"), 25),
        num_replicas=_as_int(g_map.get("num_replicas"), 30),
        nivel_confianza=_as_float(g_map.get("nivel_confianza"), 0.95),
        num_balanzas=_as_int(g_map.get("num_balanzas"), 1),
        cap_descarga_mineral=_as_int(g_map.get("cap_descarga_mineral"), 0),
        cap_descarga_desmonte=_as_int(g_map.get("cap_descarga_desmonte"), 0),
        dist_balanza=_as_str(g_map.get("dist_balanza"), "TBALANZA"),
        dist_descarga_mineral=_as_str(g_map.get("dist_descarga_mineral"), "TMIN"),
        dist_descarga_desmonte=_as_str(g_map.get("dist_descarga_desmonte"), "TDESM"),
        dist_tonelaje_mineral=_as_str(g_map.get("dist_tonelaje_mineral"), "TN"),
        costo_op_volquete_hr=_as_float(g_map.get("costo_op_volquete_hr"), 50.0),
        costo_cola_volquete_hr=_as_float(g_map.get("costo_cola_volquete_hr"), 30.0),
        costo_op_scoop_hr=_as_float(g_map.get("costo_op_scoop_hr"), 160.0),
        costo_improd_scoop_hr=_as_float(g_map.get("costo_improd_scoop_hr"), 90.0),
        usar_tope_planta=_as_bool(g_map.get("usar_tope_planta"), False),
        cap_planta_tn=_as_float(g_map.get("cap_planta_tn"), 0.0),
        flota_min=_as_int(g_map.get("flota_min"), 10),
        flota_max=_as_int(g_map.get("flota_max"), 18),
        funcion_objetivo=_as_str(g_map.get("funcion_objetivo"), "INGRESO_NETO").upper(),
        hp_volquete=_as_float(g_map.get("hp_volquete"), 400.0),
        dm_volquete=_as_float(g_map.get("dm_volquete"), 0.933),
        hp_scoop=_as_float(g_map.get("hp_scoop"), 250.0),
        dm_scoop=_as_float(g_map.get("dm_scoop"), 0.85),
        factor_vent_m3_min_hp=_as_float(g_map.get("factor_vent_m3_min_hp"), 3.0),
        diesel_ralenti_l_hr=_as_float(g_map.get("diesel_ralenti_l_hr"), 3.5),
        precio_diesel_usd_l=_as_float(g_map.get("precio_diesel_usd_l"), 1.20),
        factor_co2_kg_l=_as_float(g_map.get("factor_co2_kg_l"), 2.68),
    )

    # 2. Distribuciones (Paramétricas + Empíricas)
    distributions: Dict[str, DistributionSpec] = {}
    if "3_Distribuciones_Parametricas" in xls.sheet_names:
        df_p = pd.read_excel(xls, sheet_name="3_Distribuciones_Parametricas")
        for _, r in df_p.iterrows():
            name = _as_str(r.get("dist_name"))
            if not name:
                continue
            distributions[name] = DistributionSpec(
                dist_type=_as_str(r.get("dist_type"), "CONSTANT"),
                param1=_as_float(r.get("param1"), 0.0),
                param2=_as_float(r.get("param2"), 0.0),
                param3=_as_float(r.get("param3"), 0.0),
                min_val=_as_float(r.get("min_val"), 0.0),
                unit=_as_str(r.get("unidad"), "s"),
                category=_as_str(r.get("notas"), "Paramétrica"),
            )

    if "4_Ojivas_Empiricas" in xls.sheet_names:
        df_e = pd.read_excel(xls, sheet_name="4_Ojivas_Empiricas")
        if not df_e.empty:
            for name, grp in df_e.groupby("dist_name", sort=False):
                pts = [(float(p), float(v)) for p, v in zip(grp["prob_acum"], grp["valor"])]
                unit = _as_str(grp["unidad"].iloc[0], "s") if "unidad" in grp else "s"
                cat = _as_str(grp["categoria"].iloc[0], "Empírica") if "categoria" in grp else "Empírica"
                distributions[str(name).strip()] = DistributionSpec(
                    dist_type="EMPIRICAL", points=pts, unit=unit, category=cat
                )

    # 3. Zonas (N dinámicas, 9 parámetros de ruta anclados en bocamina)
    df_z = pd.read_excel(xls, sheet_name="2_Zonas")
    zones: List[ZoneConfig] = []
    for _, r in df_z.iterrows():
        zid = _as_str(r.get("zone_id"))
        if not zid:
            continue
        waste_raw = r.get("waste_tonnage_dist", 20.0)
        try:
            waste_ref: Union[str, float] = float(waste_raw)
            if math.isnan(waste_ref):
                waste_ref = 20.0
        except (TypeError, ValueError):
            waste_ref = _as_str(waste_raw, "20.0")

        ore_dist_override = _as_str(r.get("ore_tonnage_dist"), "")
        hp_raw = r.get("hp_scoop")
        hp_val = None if (pd.isna(hp_raw) or str(hp_raw).strip() == "") else float(hp_raw)

        zones.append(
            ZoneConfig(
                zone_id=zid,
                zone_name=_as_str(r.get("zone_name"), zid),
                active=_as_bool(r.get("active"), True),
                n_trucks=_as_int(r.get("n_trucks"), 1),
                n_scoops=_as_int(r.get("n_scoops"), 1),
                prob_mineral=_as_float(r.get("prob_mineral"), 0.75),
                quota_mineral_tn=_as_float(r.get("quota_mineral_tn"), 0.0),
                max_quota_mineral_tn=_as_float(r.get("max_quota_mineral_tn"), 0.0),
                ore_price_usd_tn=_as_float(r.get("ore_price_usd_tn"), 60.0),
                waste_tonnage_dist=waste_ref,
                ore_tonnage_dist=ore_dist_override or None,
                entry_start_sec=_as_float(r.get("entry_start_sec"), 900.0),
                entry_interval_sec=_as_float(r.get("entry_interval_sec"), 240.0),
                dist_inicial=_parse_chain(r.get("dist_inicial")),
                dist_bajada_vacio=_parse_chain(r.get("dist_bajada_vacio")),
                dist_carguio=_as_str(r.get("dist_carguio"), "0"),
                dist_subida_cargado=_parse_chain(r.get("dist_subida_cargado")),
                dist_hacia_balanza=_parse_chain(r.get("dist_hacia_balanza")),
                dist_balanza_a_cancha=_parse_chain(r.get("dist_balanza_a_cancha")),
                dist_hacia_desmontera=_parse_chain(r.get("dist_hacia_desmontera")),
                dist_retorno_desde_mineral=_parse_chain(r.get("dist_retorno_desde_mineral")),
                dist_retorno_desde_desmonte=_parse_chain(r.get("dist_retorno_desde_desmonte")),
                facility_name=_as_str(r.get("facility_name"), ""),
                queue_name=_as_str(r.get("queue_name"), ""),
                hp_scoop=hp_val,
                cap_max_ventilacion_m3_min=_as_float(r.get("cap_max_ventilacion_m3_min"), 0.0),
            )
        )

    # Sincronización automática de la meta diaria con la suma de cuotas de zonas activas
    active_quota_sum = sum(float(z.quota_mineral_tn) for z in zones if z.active)
    if active_quota_sum > 0.0:
        global_cfg.meta_mineral_gdia = active_quota_sum

    # 4. Contingencias / Fallas
    breakdowns: List[BreakdownConfig] = []
    if "5_Contingencias_Fallas" in xls.sheet_names:
        df_b = pd.read_excel(xls, sheet_name="5_Contingencias_Fallas")
        for _, r in df_b.iterrows():
            eid = _as_str(r.get("event_id"))
            skey = _as_str(r.get("station_key"))
            if not eid or not skey:
                continue
            breakdowns.append(
                BreakdownConfig(
                    event_id=eid,
                    station_key=skey,
                    mode=_as_str(r.get("mode"), "SCHEDULED"),
                    start_min=_as_float(r.get("start_min"), 0.0),
                    duration_min=_as_float(r.get("duration_min"), 0.0),
                    mtbf_hr=_as_float(r.get("mtbf_hr"), 0.0),
                    mttr_min=_as_float(r.get("mttr_min"), 0.0),
                    units_affected=_as_int(r.get("units_affected"), 1),
                    active=_as_bool(r.get("active"), False),
                )
            )

    return SimulationConfig(
        global_cfg=global_cfg,
        zones=zones,
        distributions=distributions,
        breakdowns=breakdowns,
    )


# ==============================================================================
# 8. BATERÍA DE AUTOVERIFICACIÓN (FÍSICA + COSTOS + VENTILACIÓN + CO2 + TRACE)
# ==============================================================================
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1600)

    print("=" * 105)
    print(" 1. VALIDACIÓN FÍSICA HISTÓRICA (12 VOLQUETES: 4-2-6, FIJA, seed=14)")
    print("=" * 105)
    cfg_val = build_huancapeti_base_config(
        n_her=4, n_car=2, n_cot=6, policy="FIJA", prob_mode="GPSS_WORLD_VAL", seed=14
    )
    eng_val = SubHaulDESEngine(cfg_val)
    kpi_val = eng_val.run()
    aprox = (1.0 - abs(1200.0 - kpi_val["ore_total_tn"]) / 1200.0) * 100.0
    print(f"Producción Mineral: {kpi_val['ore_total_tn']:.2f} t/gdia | Aproximación vs 1,200 t: {aprox:.2f}%")
    assert abs(kpi_val["ore_total_tn"] - 1214.19) < 0.01, f"Paridad histórica alterada: {kpi_val['ore_total_tn']:.4f}"
    assert aprox > 97.0

    print("\n" + "=" * 105)
    print(" 2. AUDITORÍA ESTRICTA DE TIEMPOS, COSTOS, VENTILACIÓN D.S. 023 Y CO2 (6-2-7, DIN_ETA_RITMO, seed=25)")
    print("=" * 105)
    cfg_aud = build_huancapeti_base_config(n_her=6, n_car=2, n_cot=7, policy="DIN_ETA_RITMO", seed=25)
    eng_aud = SubHaulDESEngine(cfg_aud)
    kpi_aud = eng_aud.run()
    df_econ = eng_aud.get_economic_dataframe()
    print(df_econ.to_string(index=False))

    df_st = eng_aud.get_stations_report_dataframe()
    assert list(df_st["Estación"])[:3] == [
        "Scooptram (Hércules)",
        "Scooptram (Caridad)",
        "Scooptram (Coturcán)",
    ], f"Etiquetas de estaciones inesperadas: {list(df_st['Estación'])}"
    assert "Balanza" in list(df_st["Estación"])

    eff_hr = eng_aud.effective_horizon / 3600.0
    print("\n--- Verificación de partición de tiempos (Scooptrams) ---")
    for z in eng_aud.active_zones:
        st = eng_aud.stations[z.zone_id]
        busy_hr = st.busy_area_time / 3600.0
        down_hr = st.downtime_area_time / 3600.0
        idle_hr = max(0.0, (st.capacity * eff_hr) - busy_hr - down_hr)
        print(
            f"  {z.zone_id}: busy={busy_hr:.4f} h + idle={idle_hr:.4f} h + down={down_hr:.4f} h "
            f"= {busy_hr + idle_hr + down_hr:.4f} h (cap*H = {st.capacity * eff_hr:.4f} h)"
        )
        assert abs((busy_hr + idle_hr + down_hr) - (st.capacity * eff_hr)) < 1e-6

    truck_prod_hr = sum(eng_aud.zone_truck_prod_sec.values()) / 3600.0
    truck_wait_hr = sum(eng_aud.zone_truck_wait_sec.values()) / 3600.0
    max_truck_hr = eng_aud.n_trucks_total * eff_hr
    print(
        f"\n--- Verificación de partición de tiempos (Volquetes) ---\n"
        f"  prod={truck_prod_hr:.4f} h + wait={truck_wait_hr:.4f} h = {truck_prod_hr + truck_wait_hr:.4f} h "
        f"<= N*H ({max_truck_hr:.4f} h)"
    )
    assert truck_prod_hr + truck_wait_hr <= max_truck_hr + 1e-6

    sum_partials = (
        kpi_aud["scoop_op_cost_usd"]
        + kpi_aud["scoop_idle_cost_usd"]
        + kpi_aud["truck_op_cost_usd"]
        + kpi_aud["truck_wait_cost_usd"]
    )
    print(
        f"\n--- Verificación de costos e ingreso neto ---\n"
        f"  Suma parciales = ${sum_partials:,.2f} | total_opex_usd = ${kpi_aud['total_opex_usd']:,.2f}\n"
        f"  gross - opex   = ${kpi_aud['gross_revenue_usd'] - kpi_aud['total_opex_usd']:,.2f} "
        f"| net_revenue_usd = ${kpi_aud['net_revenue_usd']:,.2f}"
    )
    assert abs(sum_partials - kpi_aud["total_opex_usd"]) < 0.02
    assert abs((kpi_aud["gross_revenue_usd"] - kpi_aud["total_opex_usd"]) - kpi_aud["net_revenue_usd"]) < 0.02
    assert cfg_aud.global_cfg.dm_volquete == 0.933
    assert cfg_aud.global_cfg.avoid_down_zones is True

    tot_row = df_econ.iloc[-1]
    assert abs(float(tot_row["Costo Total (US$)"]) - kpi_aud["total_opex_usd"]) < 0.01
    assert abs(float(tot_row["Ingreso Neto (US$)"]) - kpi_aud["net_revenue_usd"]) < 0.01
    assert abs((kpi_aud["truck_op_cost_usd"] + kpi_aud["truck_wait_cost_usd"]) - kpi_aud["fleet_op_cost_usd"]) < 0.02

    # Verificación exacta de Q_pico por zona (Fu=1.0, sin Dm)
    for z in eng_aud.active_zones:
        zid = z.zone_id
        peak = kpi_aud["underground_trucks_peak_by_zone"][zid]
        expected_q = round(3.0 * (z.n_scoops * 250.0 + peak * 400.0), 2)
        assert abs(kpi_aud["vent_q_peak_m3_min_by_zone"][zid] - expected_q) < 0.01

    print(
        f"\n--- Verificación Ventilación D.S. 023-2017-EM (Pico Interior Mina, Fu=1.0, sin Dm) ---\n"
        f"  Picos volquetes int. : {kpi_aud['underground_trucks_peak_by_zone']}\n"
        f"  Prom. volquetes int. : {kpi_aud['underground_trucks_avg_by_zone']}\n"
        f"  Q_pico zona (m3/min) : {kpi_aud['vent_q_peak_m3_min_by_zone']}\n"
        f"  Q_pico zona (CFM)    : {kpi_aud['vent_q_peak_cfm_by_zone']}\n"
        f"  Q_total (m3/min)     : {kpi_aud['vent_q_peak_total_m3_min']} ({kpi_aud['vent_q_peak_total_cfm']} CFM)"
    )
    print(
        f"\n--- Verificación Diésel y CO2 en Colas de Volquetes ---\n"
        f"  Diésel : {kpi_aud['diesel_wasted_wait_l_shift']} L/gdia | {kpi_aud['diesel_wasted_wait_l_year']} L/año\n"
        f"  Costo  : US$ {kpi_aud['diesel_wasted_wait_cost_usd_shift']}/gdia | US$ {kpi_aud['diesel_wasted_wait_cost_usd_year']}/año\n"
        f"  CO2    : {kpi_aud['co2_wasted_wait_kg_shift']} kg/gdia | {kpi_aud['co2_wasted_wait_t_year']} t/año"
    )
    print("OK: Auditoría de tiempos, costos, ventilación D.S. 023 y CO2 aprobada.")

    print("\n" + "=" * 105)
    print(" 3. IDA Y VUELTA PLANTILLA EXCEL (.xlsx) + TOPE EXACTO DE PLANTA (24 VOLQ: 6-9-9, FIJA)")
    print("=" * 105)
    tpl_path = export_excel_template("plantilla_subhaul_opt_base.xlsx", cfg_aud)
    cfg_reloaded = load_config_from_excel(tpl_path)
    kpi_reloaded = SubHaulDESEngine(cfg_reloaded).run()
    assert abs(kpi_reloaded["ore_total_tn"] - kpi_aud["ore_total_tn"]) < 1e-6
    assert abs(kpi_reloaded["net_revenue_usd"] - kpi_aud["net_revenue_usd"]) < 1e-6

    cfg_cap = build_huancapeti_base_config(n_her=6, n_car=9, n_cot=9, policy="FIJA", seed=25)
    cfg_cap.global_cfg.usar_tope_planta = True
    cfg_cap.global_cfg.cap_planta_tn = 1500.0
    kpi_cap = SubHaulDESEngine(cfg_cap).run()
    assert kpi_cap["ore_total_tn"] > 1500.0
    assert abs(kpi_cap["capped_ore_tn"] - 1500.0) < 1e-6, (
        f"Tope exacto no respetado: capped_ore_tn={kpi_cap['capped_ore_tn']}"
    )
    assert abs(kpi_cap["net_revenue_usd"] - (kpi_cap["gross_revenue_usd"] - kpi_cap["total_opex_usd"])) < 0.02
    print(
        f"OK: Excel y tope exacto de planta validados "
        f"(físico={kpi_cap['ore_total_tn']:.2f} t | remunerado={kpi_cap['capped_ore_tn']:.2f} t)."
    )

    print("\n" + "=" * 105)
    print(" 4. PARIDAD DE TRACE (record_trace=True vs False) Y SERIALIZACIÓN JSON")
    print("=" * 105)
    eng_val_trace = SubHaulDESEngine(cfg_val.clone(), record_trace=True)
    kpi_val_trace = eng_val_trace.run()
    assert kpi_val_trace["ore_total_tn"] == kpi_val["ore_total_tn"], (
        f"Paridad ore_total_tn alterada (FIJA): {kpi_val_trace['ore_total_tn']} vs {kpi_val['ore_total_tn']}"
    )
    assert kpi_val_trace["net_revenue_usd"] == kpi_val["net_revenue_usd"], (
        f"Paridad net_revenue_usd alterada (FIJA): {kpi_val_trace['net_revenue_usd']} vs {kpi_val['net_revenue_usd']}"
    )
    assert len(eng_val_trace.trace["redirections"]) == 0, "Política FIJA no debe generar redirecciones."

    eng_aud_trace = SubHaulDESEngine(cfg_aud.clone(), record_trace=True)
    kpi_aud_trace = eng_aud_trace.run()
    assert kpi_aud_trace["ore_total_tn"] == kpi_aud["ore_total_tn"], (
        f"Paridad ore_total_tn alterada (DIN): {kpi_aud_trace['ore_total_tn']} vs {kpi_aud['ore_total_tn']}"
    )
    assert kpi_aud_trace["net_revenue_usd"] == kpi_aud["net_revenue_usd"], (
        f"Paridad net_revenue_usd alterada (DIN): {kpi_aud_trace['net_revenue_usd']} vs {kpi_aud['net_revenue_usd']}"
    )

    payload = eng_aud_trace.get_trace_payload()
    payload_json = json.dumps(payload, ensure_ascii=False)
    assert len(payload["travels"]) > 0
    assert len(payload["stations"]) > 0
    assert len(payload["dumps"]) > 0
    assert len(payload["redirections"]) > 0
    print(
        f"Paridad 100% verificada (FIJA={kpi_val_trace['ore_total_tn']:.2f} t | "
        f"DIN={kpi_aud_trace['ore_total_tn']:.2f} t, Neto=${kpi_aud_trace['net_revenue_usd']:,.2f})"
    )
    print(
        f"Eventos grabados (DIN_ETA_RITMO, seed=25) | JSON: {len(payload_json) / 1024.0:.1f} KB\n"
        f"  - travels     : {len(payload['travels'])}\n"
        f"  - stations    : {len(payload['stations'])}\n"
        f"  - dumps       : {len(payload['dumps'])}\n"
        f"  - breakdowns  : {len(payload['breakdowns'])}\n"
        f"  - redirections: {len(payload['redirections'])}"
    )
    print("OK: Trace 2D validado y serializable a JSON sin errores.")