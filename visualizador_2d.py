"""
SubHaul-Opt — Visualizador 2D Interactivo (HTML5 Canvas + Vanilla JS).
Feria ABET UNI 2026-II | Métodos de Explotación Subterránea (MI753-A)

Genera componentes HTML autocontenidos (cero dependencias externas, aptos para
`st.components.v1.html` y uso offline) a partir del diccionario entregado por
`SubHaulDESEngine.get_trace_payload()`.

Modos de layout automáticos:
  A. Modo Huancapetí (cuando las zonas activas son exactamente {HER, CAR, COT}):
     Replica la geometría, curvas, tablero HUD y paleta de la Figura 35.
  B. Modo Genérico (N zonas):
     Layout columnar limpio de izquierda a derecha con rampas proporcionales a
     `exp_bajada_sec` y paleta cromática automática.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List


def _safe_json_for_script(obj: Any) -> str:
    """Serializa a JSON escapando cierres de etiqueta script para incrustar en HTML."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/")


def _render_2d_html(
    panes: List[Dict[str, Any]],
    height: int,
    is_comparison: bool,
) -> str:
    """Construye el documento HTML5 autocontenido para vista individual o comparativa."""
    uid = f"sh2d_{uuid.uuid4().hex[:10]}"
    data_json = _safe_json_for_script(
        {
            "uid": uid,
            "is_comparison": is_comparison,
            "panes": panes,
        }
    )

    has_any_breakdown = any(bool(p.get("breakdowns")) for p in panes)
    delta_bar_h = 34 if is_comparison else 0
    controls_h = 56
    canvas_area_h = max(320, int(height) - controls_h - delta_bar_h - 16)

    delta_bar_html = (
        f"""
        <div class="sh-delta-bar" id="{uid}_delta">
          <span class="sh-delta-pill" id="{uid}_d_ore">Δ Mineral: --</span>
          <span class="sh-delta-pill" id="{uid}_d_waste">Δ Desmonte: --</span>
          <span class="sh-delta-pill" id="{uid}_d_redir">Redirecciones: --</span>
          <span class="sh-delta-pill" id="{uid}_d_queue">Colas activas: --</span>
        </div>
        """
        if is_comparison
        else ""
    )

    if is_comparison:
        canvases_html = f"""
        <div class="sh-grid-compare">
          <div class="sh-pane">
            <div class="sh-pane-hdr" id="{uid}_hdr_0"></div>
            <div class="sh-canvas-wrap" style="height:{canvas_area_h}px;">
              <canvas id="{uid}_cv_0"></canvas>
              <div class="sh-tooltip" id="{uid}_tt_0"></div>
            </div>
          </div>
          <div class="sh-pane">
            <div class="sh-pane-hdr" id="{uid}_hdr_1"></div>
            <div class="sh-canvas-wrap" style="height:{canvas_area_h}px;">
              <canvas id="{uid}_cv_1"></canvas>
              <div class="sh-tooltip" id="{uid}_tt_1"></div>
            </div>
          </div>
        </div>
        """
    else:
        canvases_html = f"""
        <div class="sh-pane sh-pane-single">
          <div class="sh-canvas-wrap" style="height:{canvas_area_h}px;">
            <canvas id="{uid}_cv_0"></canvas>
            <div class="sh-tooltip" id="{uid}_tt_0"></div>
          </div>
        </div>
        """

    fault_btn_html = (
        f'<button class="sh-btn sh-btn-fault" id="{uid}_fault" title="Saltar justo antes de la primera falla">🚨 Ir a falla</button>'
        if has_any_breakdown
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SubHaul-Opt — Simulación 2D</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #14181f;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
    color: #e5e7eb;
    user-select: none;
  }}
  .sh-root {{
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px;
    background: #14181f;
  }}
  .sh-delta-bar {{
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    background: #1d2430;
    border: 1px solid #303a4b;
    border-radius: 8px;
    padding: 0 10px;
    font-size: 12px;
    font-weight: 600;
  }}
  .sh-delta-pill {{
    background: #11161f;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 2px 10px;
    color: #e2e8f0;
    white-space: nowrap;
  }}
  .sh-grid-compare {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    flex: 1;
    min-height: 0;
  }}
  @media (max-width: 820px) {{
    .sh-grid-compare {{ grid-template-columns: 1fr; }}
  }}
  .sh-pane {{
    display: flex;
    flex-direction: column;
    background: #1b212c;
    border: 1px solid #2e3848;
    border-radius: 10px;
    overflow: hidden;
    min-height: 0;
  }}
  .sh-pane-single {{ flex: 1; }}
  .sh-pane-hdr {{
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 10px;
    background: #11161f;
    border-bottom: 1px solid #2b3444;
    font-size: 12px;
    font-weight: 700;
    color: #f8fafc;
  }}
  .sh-canvas-wrap {{
    position: relative;
    width: 100%;
    flex: 1;
    min-height: 240px;
    background: #cfc49b;
    overflow: hidden;
  }}
  .sh-canvas-wrap canvas {{
    display: block;
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }}
  .sh-tooltip {{
    position: absolute;
    pointer-events: none;
    display: none;
    background: rgba(15, 23, 42, 0.94);
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 11px;
    line-height: 1.38;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    z-index: 20;
    white-space: nowrap;
  }}
  .sh-controls {{
    min-height: 50px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    background: #1b222e;
    border: 1px solid #303b4d;
    border-radius: 10px;
    padding: 6px 12px;
  }}
  .sh-btn {{
    background: #273244;
    color: #f1f5f9;
    border: 1px solid #3d4d66;
    border-radius: 7px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, transform 0.08s;
    white-space: nowrap;
  }}
  .sh-btn:hover {{ background: #334155; }}
  .sh-btn:active {{ transform: scale(0.97); }}
  .sh-btn-primary {{
    background: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
  }}
  .sh-btn-primary:hover {{ background: #1d4ed8; }}
  .sh-btn-fault {{
    background: #991b1b;
    border-color: #ef4444;
    color: #fef2f2;
  }}
  .sh-btn-fault:hover {{ background: #b91c1c; }}
  .sh-speed-group {{
    display: inline-flex;
    align-items: center;
    background: #11161f;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
  }}
  .sh-spd {{
    background: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 5px;
    padding: 3px 7px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }}
  .sh-spd:hover {{ color: #f8fafc; }}
  .sh-spd.active {{
    background: #0ea5e9;
    color: #082f49;
  }}
  .sh-scrubber-wrap {{
    flex: 1;
    min-width: 140px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .sh-scrubber {{
    flex: 1;
    accent-color: #38bdf8;
    cursor: pointer;
    height: 6px;
  }}
  .sh-clock {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    color: #f8fafc;
    background: #11161f;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 3px 8px;
    min-width: 132px;
    text-align: center;
    white-space: nowrap;
  }}
</style>
</head>
<body>
<div class="sh-root" id="{uid}_root">
  {delta_bar_html}
  {canvases_html}
  <div class="sh-controls">
    <button class="sh-btn sh-btn-primary" id="{uid}_play">⏸️ Pausa</button>
    <button class="sh-btn" id="{uid}_reset">⏮️ Reiniciar</button>
    <div class="sh-speed-group" id="{uid}_speeds">
      <button class="sh-spd" data-spd="10">10x</button>
      <button class="sh-spd" data-spd="30">30x</button>
      <button class="sh-spd active" data-spd="60">60x</button>
      <button class="sh-spd" data-spd="180">180x</button>
      <button class="sh-spd" data-spd="300">300x</button>
    </div>
    <div class="sh-scrubber-wrap">
      <input type="range" class="sh-scrubber" id="{uid}_scrub" min="0" max="32400" step="1" value="0" />
      <div class="sh-clock" id="{uid}_clock">00:00:00</div>
    </div>
    {fault_btn_html}
  </div>
</div>

<script>
(function() {{
  const CONFIG = {data_json};
  const UID = CONFIG.uid;

  const ZONE_PALETTE = [
    "#FACC15", "#38BDF8", "#22C55E", "#F97316",
    "#A78BFA", "#FB7185", "#2DD4BF", "#E879F9"
  ];
  const FIXED_ZONE_COLORS = {{
    "HER": "#FACC15",
    "CAR": "#38BDF8",
    "COT": "#22C55E"
  }};

  function formatHMS(sec) {{
    const s = Math.max(0, Math.floor(sec));
    const hh = String(Math.floor(s / 3600)).padStart(2, "0");
    const mm = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }}

  function fmtTruckId(id) {{
    return "V" + String(id).padStart(2, "0");
  }}

  // --------------------------------------------------------------------------
  // Utilidades geométricas (Catmull-Rom suavizado + parametrización por arco)
  // --------------------------------------------------------------------------
  function catmullRomPoints(ctrl, stepsPerSeg) {{
    if (!ctrl || ctrl.length < 2) return ctrl || [];
    if (ctrl.length === 2) {{
      const out = [];
      const n = Math.max(2, stepsPerSeg);
      for (let i = 0; i <= n; i++) {{
        const u = i / n;
        out.push([
          ctrl[0][0] + (ctrl[1][0] - ctrl[0][0]) * u,
          ctrl[0][1] + (ctrl[1][1] - ctrl[0][1]) * u
        ]);
      }}
      return out;
    }}
    const pts = [];
    const n = ctrl.length;
    for (let i = 0; i < n - 1; i++) {{
      const p0 = ctrl[Math.max(0, i - 1)];
      const p1 = ctrl[i];
      const p2 = ctrl[i + 1];
      const p3 = ctrl[Math.min(n - 1, i + 2)];
      for (let s = 0; s < stepsPerSeg; s++) {{
        const t = s / stepsPerSeg;
        const t2 = t * t;
        const t3 = t2 * t;
        const x = 0.5 * (
          (2 * p1[0]) +
          (-p0[0] + p2[0]) * t +
          (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
          (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
        );
        const y = 0.5 * (
          (2 * p1[1]) +
          (-p0[1] + p2[1]) * t +
          (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
          (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
        );
        pts.push([x, y]);
      }}
    }}
    pts.push([ctrl[n - 1][0], ctrl[n - 1][1]]);
    return pts;
  }}

  function buildPolyline(ctrlPoints, smooth) {{
    const pts = smooth ? catmullRomPoints(ctrlPoints, 16) : ctrlPoints.slice();
    const cum = [0];
    let total = 0;
    for (let i = 1; i < pts.length; i++) {{
      const dx = pts[i][0] - pts[i - 1][0];
      const dy = pts[i][1] - pts[i - 1][1];
      total += Math.hypot(dx, dy);
      cum.push(total);
    }}
    return {{ pts, cum, total: Math.max(total, 1e-6) }};
  }}

  function samplePolyline(poly, u, laneOffset) {{
    const uc = Math.max(0, Math.min(1, u));
    const target = uc * poly.total;
    const pts = poly.pts;
    const cum = poly.cum;
    let lo = 0, hi = cum.length - 1;
    while (lo < hi - 1) {{
      const mid = (lo + hi) >> 1;
      if (cum[mid] <= target) lo = mid;
      else hi = mid;
    }}
    const segLen = Math.max(1e-6, cum[hi] - cum[lo]);
    const frac = (target - cum[lo]) / segLen;
    const x0 = pts[lo][0], y0 = pts[lo][1];
    const x1 = pts[hi][0], y1 = pts[hi][1];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const ang = Math.atan2(dy, dx);
    const len = Math.max(1e-6, Math.hypot(dx, dy));
    // Normal derecha respecto al sentido de avance
    const nx = -dy / len;
    const ny = dx / len;
    const off = laneOffset || 0;
    return {{
      x: x0 + dx * frac + nx * off,
      y: y0 + dy * frac + ny * off,
      angle: ang
    }};
  }}

  // --------------------------------------------------------------------------
  // Construcción del layout espacial (A: Huancapetí Fig. 35 | B: Genérico N)
  // --------------------------------------------------------------------------
  function isHuancapetiLayout(payload) {{
    const ids = (payload.zones || []).map(z => String(z.id).toUpperCase()).sort();
    return ids.length === 3 && ids[0] === "CAR" && ids[1] === "COT" && ids[2] === "HER";
  }}

  function buildHuancapetiLayout(payload) {{
    const W = 1240;
    const H = 680;
    const hasScale = (payload.num_balanzas || 0) > 0;

    const nodes = {{
      ESTACIONAMIENTO: {{ x: 162, y: 462, label: "ESTACIONAMIENTO", type: "parking" }},
      BM_HER:          {{ x: 200, y: 318, label: "BM HÉRCULES", type: "portal", zoneId: "HER" }},
      ZONA_HER:        {{ x: 356, y: 145, label: "ZONA HÉRCULES", type: "chamber", zoneId: "HER" }},
      BM_COT:          {{ x: 404, y: 402, label: "BOCAMINA COTURCÁN", type: "portal", zoneId: "COT" }},
      ZONA_COT:        {{ x: 410, y: 598, label: "ZONA COTURCÁN", type: "chamber", zoneId: "COT" }},
      CRUCE_CAR:       {{ x: 646, y: 488, label: "CRUCE CARIDAD", type: "junction" }},
      DESMONTERA:      {{ x: 596, y: 368, label: "DESMONTERA", type: "waste" }},
      BM_CAR:          {{ x: 716, y: 422, label: "BM CARIDAD", type: "portal", zoneId: "CAR" }},
      ZONA_CAR:        {{ x: 822, y: 294, label: "ZONA CARIDAD", type: "chamber", zoneId: "CAR" }},
      BALANZA:         {{ x: 1004, y: 472, label: "BALANZA", type: "scale" }},
      CANCHA_MIN:      {{ x: 1048, y: 376, label: "CANCHA-03 MINERAL", type: "ore" }}
    }};

    // Rutas exactas de la Figura 35
    const segs = {{}};
    const addSeg = (key, ctrl, underground, smooth, lane) => {{
      segs[key] = {{
        key,
        poly: buildPolyline(ctrl, smooth),
        underground: !!underground,
        lane: lane !== undefined ? lane : 5.5
      }};
    }};

    // Iniciales
    addSeg("T1", [
      [162, 434], [104, 438], [200, 318]
    ], false, false, 4.5);

    addSeg("T1KC", [
      [162, 490], [166, 560], [275, 562], [352, 514], [404, 402]
    ], false, true, 4.5);

    // Interior mina (líneas blancas)
    addSeg("T2",  [[200, 318], [356, 145]], true, false, 6.5);
    addSeg("T3",  [[356, 145], [200, 318]], true, false, 6.5);

    addSeg("T2C", [[404, 402], [410, 598]], true, false, 6.5);
    addSeg("T3C", [[410, 598], [404, 402]], true, false, 6.5);

    addSeg("T3K", [[716, 422], [822, 294]], true, false, 6.5);
    addSeg("T4K", [[822, 294], [716, 422]], true, false, 6.5);

    // Superficie entre bocaminas y desvíos
    addSeg("T4", [
      [200, 318], [240, 444], [326, 446], [404, 402]
    ], false, true, 5.5);
    addSeg("T8", [
      [404, 402], [326, 446], [240, 444], [200, 318]
    ], false, true, 5.5);

    // Desvío Coturcán
    addSeg("T4HKC", [
      [404, 402], [478, 448], [500, 516], [566, 512], [646, 488]
    ], false, true, 5.5);
    addSeg("T8HKC", [
      [646, 488], [566, 512], [500, 516], [478, 448], [404, 402]
    ], false, true, 5.5);

    // Enlace Cruce Caridad <-> BM Caridad
    addSeg("T2K", [[646, 488], [716, 422]], false, false, 5.0);
    addSeg("T7K", [[716, 422], [646, 488]], false, false, 5.0);

    // Desmontera
    addSeg("T5",  [[646, 488], [596, 368]], false, false, 5.0);
    addSeg("T6",  [[596, 368], [646, 488]], false, false, 5.0);
    addSeg("T5K", [[716, 422], [596, 368]], false, false, 5.0);
    addSeg("T6K", [[596, 368], [716, 422]], false, false, 5.0);

    // Curva doble icónica hacia Balanza y Cancha-03 Mineral
    if (hasScale) {{
      addSeg("T9", [
        [646, 488], [662, 558], [752, 608], [848, 594], [898, 514], [954, 488], [1004, 472]
      ], false, true, 3.5);
      addSeg("T10", [
        [1004, 472], [1034, 446], [1048, 376]
      ], false, true, 3.5);
      addSeg("T11", [
        [1048, 376], [1018, 434], [952, 462], [886, 486], [836, 566], [754, 580], [682, 536], [646, 488]
      ], false, true, 3.5);
    }} else {{
      addSeg("T9", [
        [646, 488], [662, 558], [752, 608], [848, 594], [898, 514], [978, 456], [1048, 376]
      ], false, true, 3.5);
      addSeg("T10", [
        [1048, 376], [1048, 376]
      ], false, false, 0);
      addSeg("T11", [
        [1048, 376], [964, 442], [886, 486], [836, 566], [754, 580], [682, 536], [646, 488]
      ], false, true, 3.5);
    }}

    // Carreteras base a dibujar (evitar duplicar trazos idénticos)
    const drawRoads = [
      {{ seg: segs["T1"],  dual: false }},
      {{ seg: segs["T1KC"], dual: false }},
      {{ seg: segs["T4"],  dual: true }},
      {{ seg: segs["T4HKC"], dual: true }},
      {{ seg: segs["T2K"], dual: true }},
      {{ seg: segs["T5"],  dual: true }},
      {{ seg: segs["T5K"], dual: true }},
      {{ seg: segs["T9"],  dual: false }},
      ...(hasScale ? [{{ seg: segs["T10"], dual: false }}] : []),
      {{ seg: segs["T11"], dual: false }},
      // Interior mina (blancas)
      {{ seg: segs["T2"],  dual: true }},
      {{ seg: segs["T2C"], dual: true }},
      {{ seg: segs["T3K"], dual: true }}
    ];

    // Bolsillos ordenados de cola por estación (posición base + vector de apilamiento)
    const queuePockets = {{
      HER:        {{ x: 302, y: 120, dx: -28, dy: 0, badgeX: 302, badgeY: 96 }},
      COT:        {{ x: 358, y: 558, dx: 0, dy: -26, badgeX: 332, badgeY: 576 }},
      CAR:        {{ x: 780, y: 266, dx: -24, dy: -16, badgeX: 780, badgeY: 238 }},
      BAL:        {{ x: 956, y: 522, dx: -26, dy: 14, badgeX: 964, badgeY: 548 }},
      DUMP_ORE:   {{ x: 1048, y: 326, dx: 0, dy: -24, badgeX: 1048, badgeY: 298 }},
      DUMP_WASTE: {{ x: 536, y: 368, dx: -26, dy: 0, badgeX: 536, badgeY: 344 }}
    }};

    // Posición del HUD por zona (inspirado fielmente en la Figura 35)
    const zoneHud = {{
      HER: {{
        barX: 432, barY: 162, barW: 34, barH: 82,
        oreX: 432, oreY: 126, wasteX: 432, wasteY: 146,
        lblX: 476, lblY: 194, nameShort: "HÉRCULES", align: "left"
      }},
      CAR: {{
        barX: 894, barY: 248, barW: 34, barH: 82,
        oreX: 734, oreY: 244, wasteX: 822, wasteY: 346,
        lblX: 938, lblY: 280, nameShort: "CARIDAD", align: "left"
      }},
      COT: {{
        barX: 484, barY: 554, barW: 34, barH: 82,
        oreX: 215, oreY: 604, wasteX: 215, wasteY: 626,
        lblX: 528, lblY: 586, nameShort: "COTURCÁN", align: "left"
      }}
    }};

    const zoneColors = {{ HER: "#FACC15", CAR: "#38BDF8", COT: "#22C55E" }};

    return {{
      mode: "HUANCAPETI",
      W, H, hasScale,
      nodes, segs, drawRoads, queuePockets, zoneHud, zoneColors
    }};
  }}

  function buildGenericLayout(payload) {{
    const zones = payload.zones || [];
    const N = Math.max(1, zones.length);
    const W = 1240;
    const H = Math.max(680, 180 + N * 135);
    const hasScale = (payload.num_balanzas || 0) > 0;

    const maxBajada = Math.max(1, ...zones.map(z => z.exp_bajada_sec || 600));
    const minBajada = Math.min(...zones.map(z => z.exp_bajada_sec || 600));

    const nodes = {{
      ESTACIONAMIENTO: {{ x: 470, y: H - 56, label: "ESTACIONAMIENTO", type: "parking" }},
      CRUCE_SUP:       {{ x: 745, y: Math.round(H * 0.52), label: "ENLACE SUPERFICIE", type: "junction" }},
      DESMONTERA:      {{ x: 865, y: 185, label: "DESMONTERA", type: "waste" }},
      BALANZA:         {{ x: 925, y: Math.round(H * 0.62), label: "BALANZA", type: "scale" }},
      CANCHA_MIN:      {{ x: 1090, y: Math.round(H * 0.52), label: "CANCHA MINERAL", type: "ore" }}
    }};

    const zoneColors = {{}};
    const zoneHud = {{}};
    const queuePockets = {{
      BAL:        {{ x: 878, y: Math.round(H * 0.62) + 34, dx: -26, dy: 0, badgeX: 878, badgeY: Math.round(H * 0.62) + 58 }},
      DUMP_ORE:   {{ x: 1090, y: Math.round(H * 0.52) - 48, dx: 0, dy: -24, badgeX: 1090, badgeY: Math.round(H * 0.52) - 74 }},
      DUMP_WASTE: {{ x: 805, y: 185, dx: -26, dy: 0, badgeX: 805, badgeY: 160 }}
    }};

    const rolePaths = {{}};
    const drawRoads = [];

    const yStart = 155;
    const yEnd = H - 135;
    const yStep = N > 1 ? (yEnd - yStart) / (N - 1) : 0;

    zones.forEach((z, idx) => {{
      const zid = String(z.id).toUpperCase();
      zoneColors[zid] = FIXED_ZONE_COLORS[zid] || ZONE_PALETTE[idx % ZONE_PALETTE.length];

      const y = N > 1 ? yStart + idx * yStep : Math.round(H * 0.48);
      const bmX = 540;
      const frac = maxBajada === minBajada
        ? 0.65
        : (z.exp_bajada_sec - minBajada) / (maxBajada - minBajada);
      const rampLen = 140 + frac * 145;
      const chX = Math.round(bmX - rampLen);

      nodes["BM_" + zid] = {{ x: bmX, y, label: "BM " + zid, type: "portal", zoneId: zid }};
      nodes["ZONA_" + zid] = {{ x: chX, y, label: z.name.toUpperCase(), type: "chamber", zoneId: zid }};

      queuePockets[zid] = {{
        x: chX + 26, y: y - 34, dx: 26, dy: 0,
        badgeX: chX + 26, badgeY: y - 56
      }};

      zoneHud[zid] = {{
        barX: 28, barY: y - 38, barW: 26, barH: 72,
        oreX: 66, oreY: y - 20, wasteX: 66, wasteY: y - 2,
        lblX: 66, lblY: y + 18, nameShort: z.name.toUpperCase(), align: "left"
      }};

      const pInit = buildPolyline([[nodes.ESTACIONAMIENTO.x, nodes.ESTACIONAMIENTO.y - 20], [bmX, y]], false);
      const pBaj  = buildPolyline([[bmX, y], [chX, y]], false);
      const pSub  = buildPolyline([[chX, y], [bmX, y]], false);
      const pBal  = hasScale
        ? buildPolyline([[bmX, y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.BALANZA.x, nodes.BALANZA.y]], true)
        : buildPolyline([[bmX, y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y]], true);
      const pBalC = hasScale
        ? buildPolyline([[nodes.BALANZA.x, nodes.BALANZA.y], [nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y]], true)
        : buildPolyline([[nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y], [nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y]], false);
      const pDesm = buildPolyline([[bmX, y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.DESMONTERA.x, nodes.DESMONTERA.y]], true);
      const pRetM = buildPolyline([[nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y - 22], [bmX, y]], true);
      const pRetD = buildPolyline([[nodes.DESMONTERA.x, nodes.DESMONTERA.y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y - 16], [bmX, y]], true);

      rolePaths[zid] = {{
        inicial:          {{ poly: pInit, underground: false, lane: 4.5 }},
        bajada:           {{ poly: pBaj,  underground: true,  lane: 6.5 }},
        subida:           {{ poly: pSub,  underground: true,  lane: 6.5 }},
        hacia_balanza:    {{ poly: pBal,  underground: false, lane: 5.0 }},
        balanza_a_cancha: {{ poly: pBalC, underground: false, lane: 4.0 }},
        hacia_desmontera: {{ poly: pDesm, underground: false, lane: 5.0 }},
        retorno_min:      {{ poly: pRetM, underground: false, lane: 5.0 }},
        retorno_desm:     {{ poly: pRetD, underground: false, lane: 5.0 }}
      }};

      drawRoads.push({{ seg: {{ poly: pInit, underground: false }}, dual: false }});
      drawRoads.push({{ seg: {{ poly: buildPolyline([[bmX, y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y]], true), underground: false }}, dual: true }});
      drawRoads.push({{ seg: {{ poly: pBaj, underground: true }}, dual: true }});
    }});

    // Troncales de superficie
    drawRoads.push({{
      seg: {{ poly: buildPolyline([[nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.DESMONTERA.x, nodes.DESMONTERA.y]], true), underground: false }},
      dual: true
    }});
    if (hasScale) {{
      drawRoads.push({{
        seg: {{ poly: buildPolyline([[nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.BALANZA.x, nodes.BALANZA.y]], true), underground: false }},
        dual: false
      }});
      drawRoads.push({{
        seg: {{ poly: buildPolyline([[nodes.BALANZA.x, nodes.BALANZA.y], [nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y]], true), underground: false }},
        dual: false
      }});
      drawRoads.push({{
        seg: {{ poly: buildPolyline([[nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y], [nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y - 22]], true), underground: false }},
        dual: false
      }});
    }} else {{
      drawRoads.push({{
        seg: {{ poly: buildPolyline([[nodes.CRUCE_SUP.x, nodes.CRUCE_SUP.y], [nodes.CANCHA_MIN.x, nodes.CANCHA_MIN.y]], true), underground: false }},
        dual: true
      }});
    }}

    return {{
      mode: "GENERIC",
      W, H, hasScale,
      nodes, segs: {{}}, rolePaths, drawRoads, queuePockets, zoneHud, zoneColors
    }};
  }}

  function resolveTravelPath(layout, ev) {{
    if (layout.mode === "HUANCAPETI") {{
      if (layout.segs[ev.seg_key]) return layout.segs[ev.seg_key];
      // Respaldo si el usuario renombró alguna clave de distribución en Huancapetí
      const zid = String(ev.zone_id || "HER").toUpperCase();
      const bm = layout.nodes["BM_" + zid] || layout.nodes.BM_COT;
      const ch = layout.nodes["ZONA_" + zid] || layout.nodes.ZONA_COT;
      const park = layout.nodes.ESTACIONAMIENTO;
      const bal = layout.hasScale ? layout.nodes.BALANZA : layout.nodes.CANCHA_MIN;
      const ore = layout.nodes.CANCHA_MIN;
      const wst = layout.nodes.DESMONTERA;
      const pairs = {{
        inicial: [park, bm, false],
        bajada: [bm, ch, true],
        subida: [ch, bm, true],
        hacia_balanza: [bm, bal, false],
        balanza_a_cancha: [bal, ore, false],
        hacia_desmontera: [bm, wst, false],
        retorno_min: [ore, bm, false],
        retorno_desm: [wst, bm, false]
      }};
      const p = pairs[ev.role] || [bm, bm, false];
      return {{ poly: buildPolyline([[p[0].x, p[0].y], [p[1].x, p[1].y]], false), underground: p[2], lane: 5.5 }};
    }}
    const zid = String(ev.zone_id || "").toUpperCase();
    const byZone = layout.rolePaths[zid];
    if (byZone && byZone[ev.role]) return byZone[ev.role];
    const firstZone = Object.keys(layout.rolePaths)[0];
    return layout.rolePaths[firstZone].bajada;
  }}

  function stationCoords(layout, stationKey) {{
    if (stationKey === "BAL") return layout.hasScale ? layout.nodes.BALANZA : layout.nodes.CANCHA_MIN;
    if (stationKey === "DUMP_ORE") return layout.nodes.CANCHA_MIN;
    if (stationKey === "DUMP_WASTE") return layout.nodes.DESMONTERA;
    return layout.nodes["ZONA_" + stationKey] || layout.nodes.ESTACIONAMIENTO;
  }}

  // --------------------------------------------------------------------------
  // Preprocesamiento de líneas de tiempo por volquete
  // --------------------------------------------------------------------------
  function buildScene(payload, canvasEl, tooltipEl, hdrEl) {{
    const layout = isHuancapetiLayout(payload)
      ? buildHuancapetiLayout(payload)
      : buildGenericLayout(payload);

    const horizon = Math.max(1, Number(payload.effective_horizon_sec || payload.sim_time_sec || 32400));
    const nTrucks = Number(payload.n_trucks_total || 0);
    const zones = payload.zones || [];
    const zoneIds = zones.map(z => String(z.id).toUpperCase());

    // Mapa inicial de zona por truck_id (mismo orden que motor_des.py)
    const initialZoneByTruck = {{}};
    let tidCounter = 1;
    zones.forEach(z => {{
      const zid = String(z.id).toUpperCase();
      for (let i = 0; i < (z.n_trucks || 0); i++) {{
        initialZoneByTruck[tidCounter++] = zid;
      }}
    }});

    const truckEvents = {{}};
    for (let id = 1; id <= nTrucks; id++) truckEvents[id] = [];

    // 1. Viajes
    const rawTravels = payload.travels || [];
    if (layout.mode === "GENERIC") {{
      // Agrupar segmentos consecutivos del mismo volquete y rol para evitar saltos en modo genérico
      const byTruck = {{}};
      rawTravels.forEach(tr => {{
        if (!byTruck[tr.truck_id]) byTruck[tr.truck_id] = [];
        byTruck[tr.truck_id].push(tr);
      }});
      Object.keys(byTruck).forEach(tidStr => {{
        const list = byTruck[tidStr].sort((a, b) => a.t0 - b.t0);
        const merged = [];
        for (let i = 0; i < list.length; i++) {{
          const cur = list[i];
          const prev = merged.length ? merged[merged.length - 1] : null;
          if (
            prev &&
            prev.role === cur.role &&
            prev.zone_id === cur.zone_id &&
            prev.load_state === cur.load_state &&
            Math.abs(cur.t0 - prev.t1) <= 0.6
          ) {{
            prev.t1 = Math.max(prev.t1, cur.t1);
            prev.seg_key = prev.seg_key + "|" + cur.seg_key;
          }} else {{
            merged.push(Object.assign({{}}, cur));
          }}
        }}
        merged.forEach(tr => {{
          if (!truckEvents[tr.truck_id]) truckEvents[tr.truck_id] = [];
          truckEvents[tr.truck_id].push({{
            kind: "TRAVEL",
            t0: tr.t0,
            t1: tr.t1,
            seg_key: tr.seg_key,
            role: tr.role,
            zone_id: String(tr.zone_id || "").toUpperCase(),
            load_state: tr.load_state || "EMPTY",
            redirected: !!tr.redirected,
            prev_zone_id: tr.prev_zone_id ? String(tr.prev_zone_id).toUpperCase() : null,
            path: resolveTravelPath(layout, tr)
          }});
        }});
      }});
    }} else {{
      rawTravels.forEach(tr => {{
        if (!truckEvents[tr.truck_id]) truckEvents[tr.truck_id] = [];
        truckEvents[tr.truck_id].push({{
          kind: "TRAVEL",
          t0: tr.t0,
          t1: tr.t1,
          seg_key: tr.seg_key,
          role: tr.role,
          zone_id: String(tr.zone_id || "").toUpperCase(),
          load_state: tr.load_state || "EMPTY",
          redirected: !!tr.redirected,
          prev_zone_id: tr.prev_zone_id ? String(tr.prev_zone_id).toUpperCase() : null,
          path: resolveTravelPath(layout, tr)
        }});
      }});
    }}

    // 2. Estaciones (colas + servicio)
    const stationSvcByZone = {{}};
    zoneIds.forEach(zid => {{ stationSvcByZone[zid] = []; }});
    const dumpStationCovered = {{}};

    (payload.stations || []).forEach(st => {{
      const tid = st.truck_id;
      if (!truckEvents[tid]) truckEvents[tid] = [];
      const skey = String(st.station_key).toUpperCase();
      const zid = String(st.zone_id || skey).toUpperCase();

      if (st.t_start_svc === null || st.t_start_svc === undefined) {{
        truckEvents[tid].push({{
          kind: "QUEUE",
          t0: st.t_enter_q,
          t1: horizon + 10,
          station_key: skey,
          zone_id: zid,
          load_state: st.load_state || "EMPTY"
        }});
      }} else {{
        if (st.t_start_svc - st.t_enter_q > 0.05) {{
          truckEvents[tid].push({{
            kind: "QUEUE",
            t0: st.t_enter_q,
            t1: st.t_start_svc,
            station_key: skey,
            zone_id: zid,
            load_state: st.load_state || "EMPTY"
          }});
        }}
        truckEvents[tid].push({{
          kind: "SERVICE",
          t0: st.t_start_svc,
          t1: st.t_end_svc,
          station_key: skey,
          zone_id: zid,
          load_state: st.load_state || "EMPTY"
        }});
        if (stationSvcByZone[skey]) {{
          stationSvcByZone[skey].push([st.t_start_svc, st.t_end_svc]);
        }}
        if (skey === "DUMP_ORE" || skey === "DUMP_WASTE") {{
          if (!dumpStationCovered[tid]) dumpStationCovered[tid] = [];
          dumpStationCovered[tid].push([st.t_start_svc, st.t_end_svc]);
        }}
      }}
    }});

    // 3. Descargas sin cola (cap_descarga == 0): agregar intervalo visual DUMP
    const dumpsSorted = (payload.dumps || []).slice().sort((a, b) => a.t - b.t);
    dumpsSorted.forEach(d => {{
      const tid = d.truck_id;
      const dur = Math.max(1, Number(d.dump_duration_sec || 30));
      const t0 = Math.max(0, d.t - dur);
      const t1 = d.t;
      const covered = (dumpStationCovered[tid] || []).some(iv => Math.abs(iv[1] - t1) < 1.5);
      if (!covered) {{
        if (!truckEvents[tid]) truckEvents[tid] = [];
        truckEvents[tid].push({{
          kind: "DUMP",
          t0, t1,
          station_key: d.material === "ORE" ? "DUMP_ORE" : "DUMP_WASTE",
          zone_id: String(d.zone_id).toUpperCase(),
          load_state: d.material === "ORE" ? "ORE" : "WASTE",
          tons: d.tons
        }});
      }}
    }});

    // Ordenar eventos de cada volquete
    const KIND_RANK = {{ TRAVEL: 1, QUEUE: 2, SERVICE: 3, DUMP: 4 }};
    Object.keys(truckEvents).forEach(tidStr => {{
      truckEvents[tidStr].sort((a, b) => (a.t0 - b.t0) || (KIND_RANK[a.kind] - KIND_RANK[b.kind]));
    }});

    const redirsSorted = (payload.redirections || []).slice().sort((a, b) => a.t - b.t);
    const breakdowns = (payload.breakdowns || []).slice().sort((a, b) => a.t_start - b.t_start);

    if (hdrEl) {{
      hdrEl.innerHTML =
        '<span>' + (payload.policy_label || payload.policy) + '</span>' +
        '<span style="color:#94a3b8;font-weight:600;">Flota: ' + nTrucks + ' volq. | Semilla: ' + payload.seed + '</span>';
    }}

    return {{
      payload,
      layout,
      horizon,
      nTrucks,
      zones,
      zoneIds,
      initialZoneByTruck,
      truckEvents,
      stationSvcByZone,
      dumpsSorted,
      redirsSorted,
      breakdowns,
      canvas: canvasEl,
      ctx: canvasEl.getContext("2d"),
      tooltip: tooltipEl,
      scale: 1,
      offX: 0,
      offY: 0,
      renderedTrucks: [],
      liveSummary: {{ ore: 0, waste: 0, redirs: 0, activeQueues: 0 }}
    }};
  }}

  // --------------------------------------------------------------------------
  // Evaluación del estado exacto de la mina en el instante simTime
  // --------------------------------------------------------------------------
  function evaluateSceneAt(scene, simTime, speed) {{
    const t = Math.max(0, Math.min(simTime, scene.horizon));
    const {{ layout, zones, zoneIds }} = scene;

    // 1. Tonelajes acumulados hasta t y popups recientes de descarga
    const oreByZone = {{}};
    const wasteByZone = {{}};
    zoneIds.forEach(z => {{ oreByZone[z] = 0; wasteByZone[z] = 0; }});
    let totalOre = 0;
    let totalWaste = 0;
    const dumpPopups = [];
    const popupWin = Math.max(25, speed * 1.4);

    for (let i = 0; i < scene.dumpsSorted.length; i++) {{
      const d = scene.dumpsSorted[i];
      if (d.t > t) break;
      const zid = String(d.zone_id).toUpperCase();
      if (d.material === "ORE") {{
        oreByZone[zid] = (oreByZone[zid] || 0) + d.tons;
        totalOre += d.tons;
      }} else {{
        wasteByZone[zid] = (wasteByZone[zid] || 0) + d.tons;
        totalWaste += d.tons;
      }}
      if (t - d.t <= popupWin) {{
        dumpPopups.push({{
          material: d.material,
          tons: d.tons,
          ageFrac: (t - d.t) / popupWin
        }});
      }}
    }}

    // 2. Redirecciones acumuladas y activas por volquete
    let totalRedirs = 0;
    const recentRedirByTruck = {{}};
    const redirWin = Math.max(45, speed * 2.4);
    for (let i = 0; i < scene.redirsSorted.length; i++) {{
      const r = scene.redirsSorted[i];
      if (r.t > t) break;
      totalRedirs++;
      if (t - r.t <= redirWin) {{
        recentRedirByTruck[r.truck_id] = {{
          from: String(r.from_zone).toUpperCase(),
          to: String(r.to_zone).toUpperCase(),
          ageFrac: (t - r.t) / redirWin
        }};
      }}
    }}

    // 3. Fallas activas en t
    const activeBreakdowns = {{}};
    for (let i = 0; i < scene.breakdowns.length; i++) {{
      const b = scene.breakdowns[i];
      if (b.t_start <= t && t < b.t_end) {{
        activeBreakdowns[String(b.station_key).toUpperCase()] = b;
      }}
    }}

    // 4. Utilización acumulada de scooptrams en t
    const utilByZone = {{}};
    zones.forEach(z => {{
      const zid = String(z.id).toUpperCase();
      const intervals = scene.stationSvcByZone[zid] || [];
      let busy = 0;
      for (let i = 0; i < intervals.length; i++) {{
        const s0 = intervals[i][0];
        const s1 = intervals[i][1];
        if (s0 >= t) continue;
        busy += Math.max(0, Math.min(t, s1) - s0);
      }}
      const denom = Math.max(1, (z.n_scoops || 1) * t);
      utilByZone[zid] = t > 0 ? Math.min(100, (busy / denom) * 100) : 0;
    }});

    // 5. Estado y posición de cada volquete
    const parkedTrucks = [];
    const queueBuckets = {{}};
    const activeTrucks = [];

    for (let tid = 1; tid <= scene.nTrucks; tid++) {{
      const evs = scene.truckEvents[tid] || [];
      const initZ = scene.initialZoneByTruck[tid] || zoneIds[0] || "HER";

      if (!evs.length || t < evs[0].t0) {{
        parkedTrucks.push({{
          truck_id: tid, zone_id: initZ, load_state: "EMPTY",
          state: "PARKED", detail: "Estacionado (esperando ingreso)"
        }});
        continue;
      }}

      // Buscar intervalo activo (priorizando el último cuyo t0 <= t)
      let active = null;
      let lastEnded = null;
      for (let i = 0; i < evs.length; i++) {{
        const ev = evs[i];
        if (ev.t0 <= t && t <= ev.t1) {{
          active = ev;
        }} else if (ev.t1 < t) {{
          lastEnded = ev;
        }} else if (ev.t0 > t) {{
          break;
        }}
      }}

      if (active) {{
        if (active.kind === "TRAVEL") {{
          const dur = Math.max(1e-6, active.t1 - active.t0);
          const u = (t - active.t0) / dur;
          const pos = samplePolyline(active.path.poly, u, active.path.lane);
          activeTrucks.push({{
            truck_id: tid,
            x: pos.x, y: pos.y, angle: pos.angle,
            zone_id: active.zone_id,
            load_state: active.load_state,
            state: "TRAVEL",
            detail: "Tramo " + active.seg_key + " (" + active.role + ")",
            redir: recentRedirByTruck[tid] || null
          }});
        }} else if (active.kind === "QUEUE") {{
          if (!queueBuckets[active.station_key]) queueBuckets[active.station_key] = [];
          queueBuckets[active.station_key].push({{
            truck_id: tid,
            t0: active.t0,
            station_key: active.station_key,
            zone_id: active.zone_id,
            load_state: active.load_state,
            waitSoFar: t - active.t0,
            redir: recentRedirByTruck[tid] || null
          }});
        }} else if (active.kind === "SERVICE" || active.kind === "DUMP") {{
          const c = stationCoords(layout, active.station_key);
          const dur = Math.max(1e-6, active.t1 - active.t0);
          const prog = Math.max(0, Math.min(1, (t - active.t0) / dur));
          activeTrucks.push({{
            truck_id: tid,
            x: c.x, y: c.y + 4, angle: 0,
            zone_id: active.zone_id,
            load_state: active.load_state,
            state: active.kind,
            progress: prog,
            detail: active.kind === "DUMP"
              ? "Descargando en " + (active.station_key === "DUMP_ORE" ? "Cancha Mineral" : "Desmontera")
              : "En servicio (" + active.station_key + ")",
            redir: recentRedirByTruck[tid] || null
          }});
        }}
      }} else if (lastEnded) {{
        // Micro-hueco de redondeo o descarga truncada al fin de guardia
        let x, y, angle = 0;
        if (lastEnded.kind === "TRAVEL") {{
          const pos = samplePolyline(lastEnded.path.poly, 1.0, lastEnded.path.lane);
          x = pos.x; y = pos.y; angle = pos.angle;
        }} else {{
          const c = stationCoords(layout, lastEnded.station_key);
          x = c.x; y = c.y + 4;
        }}
        activeTrucks.push({{
          truck_id: tid,
          x, y, angle,
          zone_id: lastEnded.zone_id,
          load_state: lastEnded.load_state,
          state: "HOLD",
          detail: "Transición / fin de guardia",
          redir: recentRedirByTruck[tid] || null
        }});
      }}
    }}

    // Acomodar volquetes estacionados en grilla ordenada dentro de ESTACIONAMIENTO
    const parkNode = layout.nodes.ESTACIONAMIENTO;
    const cols = 4;
    parkedTrucks.forEach((pk, idx) => {{
      const col = idx % cols;
      const row = Math.floor(idx / cols);
      activeTrucks.push({{
        truck_id: pk.truck_id,
        x: parkNode.x - 36 + col * 24,
        y: parkNode.y - 12 + row * 16,
        angle: 0,
        zone_id: pk.zone_id,
        load_state: "EMPTY",
        state: "PARKED",
        detail: pk.detail,
        redir: null
      }});
    }});

    // Acomodar colas en orden FIFO dentro del bolsillo de cada estación
    let totalQueuedNow = 0;
    const queueCounts = {{}};
    Object.keys(queueBuckets).forEach(skey => {{
      const qList = queueBuckets[skey].sort((a, b) => (a.t0 - b.t0) || (a.truck_id - b.truck_id));
      queueCounts[skey] = qList.length;
      totalQueuedNow += qList.length;
      const pocket = layout.queuePockets[skey] || {{ x: 100, y: 100, dx: 26, dy: 0 }};
      qList.forEach((item, idx) => {{
        const wrapRow = Math.floor(idx / 5);
        const slot = idx % 5;
        const perpX = pocket.dy !== 0 ? -22 * wrapRow : 0;
        const perpY = pocket.dx !== 0 ? -20 * wrapRow : 0;
        activeTrucks.push({{
          truck_id: item.truck_id,
          x: pocket.x + pocket.dx * slot + perpX,
          y: pocket.y + pocket.dy * slot + perpY,
          angle: 0,
          zone_id: item.zone_id,
          load_state: item.load_state,
          state: "QUEUE",
          detail: "En cola " + skey + " (" + Math.round(item.waitSoFar) + " s)",
          redir: item.redir
        }});
      }});
    }});

    scene.liveSummary = {{
      ore: totalOre,
      waste: totalWaste,
      redirs: totalRedirs,
      activeQueues: totalQueuedNow
    }};

    return {{
      t,
      oreByZone,
      wasteByZone,
      totalOre,
      totalWaste,
      totalRedirs,
      activeBreakdowns,
      utilByZone,
      activeTrucks,
      queueCounts,
      dumpPopups
    }};
  }}

  // --------------------------------------------------------------------------
  // Motor de dibujo 2D (Canvas HTML5 nítido con estilo técnico Figura 35)
  // --------------------------------------------------------------------------
  function strokePolyline(ctx, pts, offset) {{
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {{
      let x = pts[i][0], y = pts[i][1];
      if (offset) {{
        const prev = pts[Math.max(0, i - 1)];
        const next = pts[Math.min(pts.length - 1, i + 1)];
        const dx = next[0] - prev[0];
        const dy = next[1] - prev[1];
        const len = Math.max(1e-6, Math.hypot(dx, dy));
        x += (-dy / len) * offset;
        y += (dx / len) * offset;
      }}
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  function drawRoundedRect(ctx, x, y, w, h, r) {{
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }}

  function drawScoopIcon(ctx, x, y, color, isDown, flashOn) {{
    ctx.save();
    ctx.translate(x, y);
    // Cámara / estación
    drawRoundedRect(ctx, -28, -18, 56, 36, 7);
    ctx.fillStyle = isDown ? (flashOn ? "#7f1d1d" : "#450a0a") : "#1e293b";
    ctx.fill();
    ctx.lineWidth = isDown ? 2.5 : 2;
    ctx.strokeStyle = isDown ? "#ef4444" : color;
    ctx.stroke();

    // Silueta estilizada de Scooptram LHD
    ctx.fillStyle = isDown ? "#fca5a5" : color;
    ctx.fillRect(-14, -6, 20, 12);
    // Cucharón delantero
    ctx.beginPath();
    ctx.moveTo(8, -8);
    ctx.lineTo(18, -10);
    ctx.lineTo(18, 10);
    ctx.lineTo(8, 8);
    ctx.closePath();
    ctx.fill();
    // Cabina
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(-10, -4, 7, 8);
    // Ruedas
    ctx.fillStyle = "#020617";
    ctx.fillRect(-12, -9, 6, 3);
    ctx.fillRect(0, -9, 6, 3);
    ctx.fillRect(-12, 6, 6, 3);
    ctx.fillRect(0, 6, 6, 3);

    if (isDown) {{
      ctx.fillStyle = flashOn ? "#ef4444" : "#facc15";
      ctx.font = "bold 9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("⚠ FALLA", 0, -22);
    }}
    ctx.restore();
  }}

  function drawPortalIcon(ctx, x, y, color) {{
    ctx.save();
    ctx.translate(x, y);
    ctx.beginPath();
    ctx.arc(0, 0, 8, Math.PI, 0, false);
    ctx.lineTo(8, 6);
    ctx.lineTo(-8, 6);
    ctx.closePath();
    ctx.fillStyle = "#1e293b";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = color || "#f8fafc";
    ctx.stroke();
    ctx.restore();
  }}

  function drawWasteDump(ctx, x, y, totalWaste) {{
    ctx.save();
    ctx.translate(x, y);
    ctx.beginPath();
    ctx.moveTo(-36, 12);
    ctx.bezierCurveTo(-40, -14, -18, -28, 4, -26);
    ctx.bezierCurveTo(28, -24, 42, -4, 34, 16);
    ctx.bezierCurveTo(22, 28, -22, 28, -36, 12);
    ctx.closePath();
    ctx.fillStyle = "#92400e";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#451a03";
    ctx.stroke();

    // Textura rocosa sutil
    ctx.fillStyle = "#b45309";
    ctx.beginPath();
    ctx.ellipse(-8, -6, 12, 7, -0.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#78350f";
    ctx.beginPath();
    ctx.ellipse(10, 6, 11, 6, 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.textAlign = "center";
    ctx.fillStyle = "#1f2937";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText("DESMONTERA", x, y - 34);
    ctx.fillStyle = "#b91c1c";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText(totalWaste.toFixed(1) + " t", x, y - 50);
    ctx.font = "bold 10px sans-serif";
    ctx.fillText("TOTAL DE DESMONTE", x, y - 64);
    ctx.restore();
  }}

  function drawScaleStation(ctx, x, y, isDown, flashOn) {{
    ctx.save();
    ctx.translate(x, y);
    drawRoundedRect(ctx, -18, -14, 36, 28, 4);
    ctx.fillStyle = isDown ? (flashOn ? "#991b1b" : "#450a0a") : "#eab308";
    ctx.fill();
    ctx.lineWidth = 2.2;
    ctx.strokeStyle = isDown ? "#ef4444" : "#422006";
    ctx.stroke();

    ctx.fillStyle = isDown ? "#fee2e2" : "#1e293b";
    ctx.fillRect(-12, -8, 24, 16);
    ctx.fillStyle = isDown ? "#ef4444" : "#fde047";
    ctx.font = "bold 8px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(isDown ? "FALLA" : "BAL", 0, 3);
    ctx.restore();
  }}

  function drawOreStockpile(ctx, x, y, totalOre, isHuancapeti) {{
    ctx.save();
    ctx.translate(x, y);
    drawRoundedRect(ctx, -26, -18, 52, 36, 6);
    ctx.fillStyle = "#1e293b";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#f59e0b";
    ctx.stroke();

    // Pila de mineral dorado
    ctx.beginPath();
    ctx.moveTo(-18, 12);
    ctx.lineTo(-4, -10);
    ctx.lineTo(8, -6);
    ctx.lineTo(18, 12);
    ctx.closePath();
    ctx.fillStyle = "#f59e0b";
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(-10, 12);
    ctx.lineTo(-2, -6);
    ctx.lineTo(6, 12);
    ctx.closePath();
    ctx.fillStyle = "#fde047";
    ctx.fill();
    ctx.restore();

    const lblX = isHuancapeti ? x + 88 : x;
    const lblY = isHuancapeti ? y - 14 : y - 48;
    ctx.save();
    ctx.textAlign = "center";
    ctx.fillStyle = "#1f2937";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(isHuancapeti ? "CANCHA-03 MINERAL" : "CANCHA MINERAL", lblX, lblY);
    ctx.fillStyle = "#1d4ed8";
    ctx.font = "bold 16px sans-serif";
    ctx.fillText(totalOre.toFixed(1) + " t", lblX, lblY + 20);
    ctx.font = "bold 10px sans-serif";
    ctx.fillText("TOTAL DE MINERAL", lblX, lblY + 34);
    ctx.restore();
  }}

  function drawParking(ctx, x, y) {{
    ctx.save();
    ctx.translate(x, y);
    drawRoundedRect(ctx, -56, -28, 112, 56, 6);
    ctx.fillStyle = "rgba(30, 41, 59, 0.22)";
    ctx.fill();
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#334155";
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ESTACIONAMIENTO", 0, -34);
    ctx.restore();
  }}

  function drawUtilizationBar(ctx, hud, color, utilPct, oreTn, wasteTn, isDown, flashOn, compact) {{
    const x = hud.barX, y = hud.barY, w = hud.barW, h = hud.barH;

    // Caja blanca con borde oscuro (estilo Figura 35)
    ctx.save();
    ctx.fillStyle = isDown ? (flashOn ? "#fee2e2" : "#fecaca") : "#f8fafc";
    ctx.strokeStyle = isDown ? "#dc2626" : "#334155";
    ctx.lineWidth = isDown ? 2.4 : 1.5;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    // Fondo interno
    const pad = 3;
    const inW = w - pad * 2;
    const inH = h - pad * 2;
    ctx.fillStyle = isDown ? "#7f1d1d" : "#cbd5e1";
    ctx.fillRect(x + pad, y + pad, inW, inH);

    // Relleno de utilización
    if (isDown) {{
      ctx.fillStyle = flashOn ? "#ef4444" : "#b91c1c";
      ctx.fillRect(x + pad, y + pad, inW, inH);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("FALLA", x + w / 2, y + h / 2 + 3);
    }} else {{
      const fillH = Math.max(0, Math.min(1, utilPct / 100)) * inH;
      ctx.fillStyle = color;
      ctx.fillRect(x + pad, y + pad + (inH - fillH), inW, fillH);
    }}

    // Marcas 100% y 0%
    ctx.fillStyle = "#b91c1c";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("100%", x + w + 4, y + 8);
    ctx.fillText("0%", x + w + 4, y + h);

    // Porcentaje y título
    ctx.fillStyle = isDown ? "#dc2626" : "#b91c1c";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(isDown ? "⚠ FALLA" : (utilPct.toFixed(1) + "%"), hud.lblX, hud.lblY);
    ctx.font = "bold 9.5px sans-serif";
    ctx.fillStyle = "#991b1b";
    ctx.fillText("UTILIZACIÓN", hud.lblX, hud.lblY + 13);
    ctx.fillText("SCOOP " + hud.nameShort, hud.lblX, hud.lblY + 24);

    // Tonelajes de la zona
    ctx.textAlign = compact ? "left" : "left";
    ctx.fillStyle = "#1d4ed8";
    ctx.font = "bold 12.5px sans-serif";
    ctx.fillText(oreTn.toFixed(1) + " t", hud.oreX, hud.oreY);
    ctx.font = "bold 9.5px sans-serif";
    ctx.fillText("MINERAL", hud.oreX + 52, hud.oreY);

    ctx.fillStyle = "#dc2626";
    ctx.font = "bold 11.5px sans-serif";
    ctx.fillText(wasteTn.toFixed(1) + " t", hud.wasteX, hud.wasteY);
    ctx.font = "bold 9px sans-serif";
    ctx.fillText("DESMONTE", hud.wasteX + 52, hud.wasteY);
    ctx.restore();
  }}

  function drawTruck(ctx, tr, zoneColor, showLabel) {{
    ctx.save();
    ctx.translate(tr.x, tr.y);

    // Destello de redirección activa
    if (tr.redir) {{
      const pulse = 1 - tr.redir.ageFrac;
      ctx.save();
      ctx.beginPath();
      ctx.arc(0, 0, 14 + (1 - pulse) * 10, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255, 255, 255, " + (0.85 * pulse).toFixed(2) + ")";
      ctx.lineWidth = 2.5;
      ctx.stroke();
      ctx.restore();
    }}

    // Anillo de progreso en servicio / descarga
    if ((tr.state === "SERVICE" || tr.state === "DUMP") && tr.progress !== undefined) {{
      ctx.save();
      ctx.beginPath();
      ctx.arc(0, 0, 15, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * tr.progress);
      ctx.strokeStyle = tr.state === "DUMP" ? "#f59e0b" : "#38bdf8";
      ctx.lineWidth = 2.6;
      ctx.stroke();
      ctx.restore();
    }}

    // Cuerpo del volquete orientado en el sentido de marcha
    ctx.save();
    ctx.rotate(tr.angle);

    // Sombra suave
    ctx.fillStyle = "rgba(15, 23, 42, 0.35)";
    drawRoundedRect(ctx, -10, -5, 21, 13, 3);
    ctx.fill();

    // Ruedas
    ctx.fillStyle = "#090d16";
    ctx.fillRect(-8, -7.5, 4.5, 2.2);
    ctx.fillRect(2.5, -7.5, 4.5, 2.2);
    ctx.fillRect(-8, 5.3, 4.5, 2.2);
    ctx.fillRect(2.5, 5.3, 4.5, 2.2);

    // Chasis / carrocería coloreada por zona
    drawRoundedRect(ctx, -10.5, -6, 21, 12, 2.5);
    ctx.fillStyle = zoneColor;
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#0f172a";
    ctx.stroke();

    // Cabina delantera (lado derecho en rotación 0)
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(5.2, -4.5, 4.2, 9);
    ctx.fillStyle = "#bae6fd";
    ctx.fillRect(7.4, -3.5, 1.5, 7);

    // Tolva: vacía vs mineral dorado vs desmonte marrón
    ctx.fillStyle = tr.load_state === "ORE"
      ? "#d97706"
      : (tr.load_state === "WASTE" ? "#78350f" : "#334155");
    ctx.fillRect(-9.2, -4.6, 13.2, 9.2);

    if (tr.load_state === "ORE") {{
      // Brillo mineral dorado en tolva
      ctx.fillStyle = "#fde047";
      ctx.beginPath();
      ctx.ellipse(-2.6, 0, 4.6, 2.8, 0, 0, Math.PI * 2);
      ctx.fill();
    }} else if (tr.load_state === "WASTE") {{
      // Roca marrón en tolva
      ctx.fillStyle = "#b45309";
      ctx.beginPath();
      ctx.ellipse(-2.6, 0, 4.4, 2.6, 0, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();

    // Etiqueta legible (siempre horizontal)
    if (showLabel) {{
      const lbl = fmtTruckId(tr.truck_id) + " [" + tr.zone_id + "]";
      ctx.font = "bold 8.5px sans-serif";
      const tw = ctx.measureText(lbl).width + 6;
      ctx.fillStyle = "rgba(15, 23, 42, 0.80)";
      drawRoundedRect(ctx, -tw / 2, -20, tw, 11, 3);
      ctx.fill();
      ctx.fillStyle = "#f8fafc";
      ctx.textAlign = "center";
      ctx.fillText(lbl, 0, -11.8);
    }}

    // Etiqueta flotante de redirección (ej. COT -> HER)
    if (tr.redir) {{
      const rTxt = "↪ " + tr.redir.from + " → " + tr.redir.to;
      ctx.font = "bold 9.5px sans-serif";
      const rw = ctx.measureText(rTxt).width + 10;
      const lift = tr.redir.ageFrac * 10;
      ctx.globalAlpha = Math.max(0.25, 1 - tr.redir.ageFrac * 0.7);
      ctx.fillStyle = "#0f172a";
      ctx.strokeStyle = zoneColor;
      ctx.lineWidth = 1.4;
      drawRoundedRect(ctx, -rw / 2, -36 - lift, rw, 14, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = zoneColor;
      ctx.textAlign = "center";
      ctx.fillText(rTxt, 0, -25.8 - lift);
      ctx.globalAlpha = 1;
    }}
    ctx.restore();
  }}

  function drawLegend(ctx, layout, zones) {{
    const lx = layout.W - 238;
    const ly = 14;
    const lh = 76 + zones.length * 19;
    const lw = 224;

    ctx.save();
    ctx.fillStyle = "rgba(248, 243, 224, 0.94)";
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1.3;
    drawRoundedRect(ctx, lx, ly, lw, lh, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 10.5px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("LEYENDA", lx + lw / 2, ly + 15);
    ctx.beginPath();
    ctx.moveTo(lx + 8, ly + 20);
    ctx.lineTo(lx + lw - 8, ly + 20);
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.stroke();

    let cy = ly + 33;
    ctx.textAlign = "left";
    zones.forEach(z => {{
      const zid = String(z.id).toUpperCase();
      const col = layout.zoneColors[zid] || "#38BDF8";
      ctx.fillStyle = col;
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.fillRect(lx + 12, cy - 8, 18, 10);
      ctx.strokeRect(lx + 12, cy - 8, 18, 10);
      ctx.fillStyle = "#1e293b";
      ctx.font = "bold 9.5px sans-serif";
      ctx.fillText("VOLQUETE ZONA " + z.name.toUpperCase(), lx + 38, cy);
      cy += 19;
    }});

    // Carga: Vacío / Mineral / Desmonte
    cy += 2;
    const loads = [
      ["#334155", "Vacío"],
      ["#f59e0b", "Mineral"],
      ["#92400e", "Desmonte"]
    ];
    let cx = lx + 12;
    loads.forEach(item => {{
      ctx.fillStyle = item[0];
      ctx.beginPath();
      ctx.arc(cx + 4, cy - 3, 4.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#1e293b";
      ctx.font = "bold 9px sans-serif";
      ctx.fillText(item[1], cx + 12, cy);
      cx += 69;
    }});
    cy += 18;

    // Trayectorias superficie vs interior mina
    ctx.strokeStyle = "#262626";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(lx + 12, cy - 3);
    ctx.lineTo(lx + 34, cy - 3);
    ctx.stroke();
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 9px sans-serif";
    ctx.fillText("TRAYECTORIA EN SUPERFICIE", lx + 40, cy);
    cy += 16;

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(lx + 12, cy - 3);
    ctx.lineTo(lx + 34, cy - 3);
    ctx.stroke();
    ctx.fillStyle = "#1e293b";
    ctx.fillText("TRAYECTORIA INTERIOR MINA", lx + 40, cy);
    ctx.restore();
  }}

  function renderScene(scene, simTime, speed, nowMs) {{
    const {{ ctx, canvas, layout, payload, zones }} = scene;
    const st = evaluateSceneAt(scene, simTime, speed);
    const flashOn = Math.floor(nowMs / 320) % 2 === 0;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 800;
    const cssH = canvas.clientHeight || 500;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {{
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }}

    const scale = Math.min(cssW / layout.W, cssH / layout.H);
    const offX = (cssW - layout.W * scale) * 0.5;
    const offY = (cssH - layout.H * scale) * 0.5;
    scene.scale = scale;
    scene.offX = offX;
    scene.offY = offY;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Fondo estilo mapa minero (Figura 35)
    ctx.fillStyle = "#cfc49b";
    ctx.fillRect(0, 0, cssW, cssH);

    ctx.save();
    ctx.translate(offX, offY);
    ctx.scale(scale, scale);

    // 1. Encabezado y tablero HUD superior
    ctx.textAlign = "left";
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 17px sans-serif";
    const title = layout.mode === "HUANCAPETI"
      ? "ANIMACIÓN DEL SISTEMA DE CARGUÍO Y TRANSPORTE — U.M. HUANCAPETÍ"
      : "ANIMACIÓN DEL SISTEMA DE CARGUÍO Y TRANSPORTE — SUBHAUL-OPT (" + zones.length + " ZONAS)";
    ctx.fillText(title, 20, 30);

    ctx.fillStyle = "#334155";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(
      "POLÍTICA: " + (payload.policy_label || payload.policy) + "  |  CORTE: " + (payload.stop_mode || "HARD"),
      20, 50
    );

    // Bloque NRO VOLQUETES / TIEMPO SIMULADO / REDIRECCIONES
    const hx = layout.W - 525;
    ctx.fillStyle = "#111827";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText("NRO DE VOLQUETES =  " + scene.nTrucks, hx, 30);
    ctx.fillText("TIEMPO SIMULADO   =  " + formatHMS(st.t) + "  (" + st.t.toFixed(1) + " s)", hx, 50);
    ctx.fillStyle = st.totalRedirs > 0 ? "#0369a1" : "#334155";
    ctx.fillText("REDIRECCIONES       =  " + st.totalRedirs, hx, 70);

    // 2. Carreteras: primero superficie (oscuras), luego interior mina (blancas)
    layout.drawRoads.forEach(item => {{
      const pts = item.seg.poly.pts;
      if (!item.seg.underground) {{
        ctx.strokeStyle = "#333333";
        ctx.lineWidth = 2.0;
        if (item.dual) {{
          strokePolyline(ctx, pts, 4.5);
          strokePolyline(ctx, pts, -4.5);
        }} else {{
          strokePolyline(ctx, pts, 0);
        }}
      }}
    }});
    layout.drawRoads.forEach(item => {{
      const pts = item.seg.poly.pts;
      if (item.seg.underground) {{
        // Sombra sutil bajo rampa blanca para máximo contraste
        ctx.strokeStyle = "rgba(30, 41, 59, 0.25)";
        ctx.lineWidth = 4.2;
        if (item.dual) {{
          strokePolyline(ctx, pts, 5.5);
          strokePolyline(ctx, pts, -5.5);
        }} else {{
          strokePolyline(ctx, pts, 0);
        }}
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.2;
        if (item.dual) {{
          strokePolyline(ctx, pts, 5.5);
          strokePolyline(ctx, pts, -5.5);
        }} else {{
          strokePolyline(ctx, pts, 0);
        }}
      }}
    }});

    // Etiquetas de tramos clave en Huancapetí
    if (layout.mode === "HUANCAPETI") {{
      ctx.save();
      ctx.fillStyle = "#1f2937";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("DESVÍO COTURCÁN", 506, 434);
      ctx.fillText("CRUCE CARIDAD", 698, 514);
      ctx.restore();
    }}

    // 3. Nodos de infraestructura
    drawParking(ctx, layout.nodes.ESTACIONAMIENTO.x, layout.nodes.ESTACIONAMIENTO.y);
    drawWasteDump(ctx, layout.nodes.DESMONTERA.x, layout.nodes.DESMONTERA.y, st.totalWaste);
    drawOreStockpile(ctx, layout.nodes.CANCHA_MIN.x, layout.nodes.CANCHA_MIN.y, st.totalOre, layout.mode === "HUANCAPETI");

    if (layout.hasScale && layout.nodes.BALANZA) {{
      const balDown = !!st.activeBreakdowns["BAL"];
      drawScaleStation(ctx, layout.nodes.BALANZA.x, layout.nodes.BALANZA.y, balDown, flashOn);
      ctx.fillStyle = "#1f2937";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("BALANZA", layout.nodes.BALANZA.x - 24, layout.nodes.BALANZA.y - 24);
    }}

    // Bocaminas, cámaras y barras HUD por zona
    zones.forEach(z => {{
      const zid = String(z.id).toUpperCase();
      const col = layout.zoneColors[zid] || "#38BDF8";
      const bm = layout.nodes["BM_" + zid];
      const ch = layout.nodes["ZONA_" + zid];
      const isDown = !!st.activeBreakdowns[zid];

      if (bm) {{
        drawPortalIcon(ctx, bm.x, bm.y, col);
        ctx.fillStyle = "#111827";
        ctx.font = "bold 11px sans-serif";
        if (layout.mode === "HUANCAPETI" && zid === "HER") {{
          ctx.textAlign = "right";
          ctx.fillText(bm.label, bm.x - 14, bm.y - 8);
        }} else if (layout.mode === "HUANCAPETI" && zid === "CAR") {{
          ctx.textAlign = "left";
          ctx.fillText(bm.label, bm.x + 14, bm.y + 8);
        }} else {{
          ctx.textAlign = "center";
          ctx.fillText(bm.label, bm.x, bm.y - 14);
        }}
      }}

      if (ch) {{
        drawScoopIcon(ctx, ch.x, ch.y, col, isDown, flashOn);
        ctx.fillStyle = "#111827";
        ctx.font = "bold 11.5px sans-serif";
        if (layout.mode === "HUANCAPETI" && zid === "HER") {{
          ctx.textAlign = "right";
          ctx.fillText(ch.label, ch.x - 36, ch.y - 4);
        }} else {{
          ctx.textAlign = "center";
          ctx.fillText(ch.label, ch.x, ch.y - 24);
        }}
      }}

      const hud = layout.zoneHud[zid];
      if (hud) {{
        drawUtilizationBar(
          ctx, hud, col,
          st.utilByZone[zid] || 0,
          st.oreByZone[zid] || 0,
          st.wasteByZone[zid] || 0,
          isDown, flashOn,
          layout.mode === "GENERIC"
        );
      }}
    }});

    // 4. Contadores de cola ("Cola: N") solo cuando N > 0
    Object.keys(st.queueCounts).forEach(skey => {{
      const n = st.queueCounts[skey];
      if (n <= 0) return;
      const pocket = layout.queuePockets[skey];
      if (!pocket) return;
      const txt = "Cola: " + n;
      ctx.save();
      ctx.font = "bold 10.5px sans-serif";
      const tw = ctx.measureText(txt).width + 12;
      ctx.fillStyle = n >= 3 ? "#dc2626" : "#d97706";
      drawRoundedRect(ctx, pocket.badgeX - tw / 2, pocket.badgeY - 10, tw, 16, 8);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(txt, pocket.badgeX, pocket.badgeY + 1.5);
      ctx.restore();
    }});

    // 5. Volquetes
    const showLabels = scene.nTrucks <= 24;
    st.activeTrucks.forEach(tr => {{
      const col = layout.zoneColors[tr.zone_id] || "#38BDF8";
      drawTruck(ctx, tr, col, showLabels);
    }});
    scene.renderedTrucks = st.activeTrucks;

    // 6. Popups flotantes de descarga (+XX.X t)
    st.dumpPopups.forEach(pop => {{
      const base = pop.material === "ORE" ? layout.nodes.CANCHA_MIN : layout.nodes.DESMONTERA;
      const py = base.y - 24 - pop.ageFrac * 24;
      ctx.save();
      ctx.globalAlpha = Math.max(0, 1 - pop.ageFrac);
      ctx.font = "bold 11.5px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = pop.material === "ORE" ? "#1d4ed8" : "#9a3412";
      ctx.fillText("+" + pop.tons.toFixed(1) + " t", base.x, py);
      ctx.restore();
    }});

    // 7. Leyenda esquina superior derecha
    drawLegend(ctx, layout, zones);

    ctx.restore();
  }}

  // --------------------------------------------------------------------------
  // Inicialización de escenas, controles sincronizados y loop 60 FPS
  // --------------------------------------------------------------------------
  const scenes = CONFIG.panes.map((p, idx) => {{
    const cv = document.getElementById(UID + "_cv_" + idx);
    const tt = document.getElementById(UID + "_tt_" + idx);
    const hdr = document.getElementById(UID + "_hdr_" + idx);
    return buildScene(p, cv, tt, hdr);
  }});

  const maxHorizon = Math.max(...scenes.map(s => s.horizon));
  let currentSimTime = 0;
  let playing = true;
  let speed = 60;
  let lastFrameMs = performance.now();

  const btnPlay = document.getElementById(UID + "_play");
  const btnReset = document.getElementById(UID + "_reset");
  const btnFault = document.getElementById(UID + "_fault");
  const scrub = document.getElementById(UID + "_scrub");
  const clockEl = document.getElementById(UID + "_clock");
  const speedWrap = document.getElementById(UID + "_speeds");

  scrub.max = String(Math.ceil(maxHorizon));

  // Tooltip interactivo al pasar el cursor sobre volquetes
  scenes.forEach(scene => {{
    scene.canvas.addEventListener("mousemove", (evt) => {{
      const rect = scene.canvas.getBoundingClientRect();
      const mx = (evt.clientX - rect.left - scene.offX) / scene.scale;
      const my = (evt.clientY - rect.top - scene.offY) / scene.scale;
      let hit = null;
      let bestDist = 18;
      for (let i = 0; i < scene.renderedTrucks.length; i++) {{
        const tr = scene.renderedTrucks[i];
        const d = Math.hypot(tr.x - mx, tr.y - my);
        if (d < bestDist) {{ bestDist = d; hit = tr; }}
      }}
      if (hit) {{
        const matTxt = hit.load_state === "ORE" ? "Mineral" : (hit.load_state === "WASTE" ? "Desmonte" : "Vacío");
        scene.tooltip.innerHTML =
          "<b>" + fmtTruckId(hit.truck_id) + "</b> · Zona: <b>" + hit.zone_id + "</b><br>" +
          "Carga: " + matTxt + "<br>" + hit.detail;
        scene.tooltip.style.display = "block";
        const left = Math.min(rect.width - 175, Math.max(8, evt.clientX - rect.left + 12));
        const top = Math.max(8, evt.clientY - rect.top - 52);
        scene.tooltip.style.left = left + "px";
        scene.tooltip.style.top = top + "px";
      }} else {{
        scene.tooltip.style.display = "none";
      }}
    }});
    scene.canvas.addEventListener("mouseleave", () => {{
      scene.tooltip.style.display = "none";
    }});
  }});

  function syncPlayButton() {{
    btnPlay.textContent = playing ? "⏸️ Pausa" : "▶️ Play";
  }}

  btnPlay.addEventListener("click", () => {{
    if (currentSimTime >= maxHorizon) currentSimTime = 0;
    playing = !playing;
    syncPlayButton();
  }});

  btnReset.addEventListener("click", () => {{
    currentSimTime = 0;
    scrub.value = "0";
    playing = true;
    syncPlayButton();
  }});

  speedWrap.addEventListener("click", (evt) => {{
    const b = evt.target.closest("button[data-spd]");
    if (!b) return;
    speed = Number(b.getAttribute("data-spd")) || 60;
    speedWrap.querySelectorAll("button").forEach(el => el.classList.remove("active"));
    b.classList.add("active");
  }});

  scrub.addEventListener("input", () => {{
    currentSimTime = Number(scrub.value) || 0;
  }});

  if (btnFault) {{
    const allBd = [];
    scenes.forEach(s => s.breakdowns.forEach(b => allBd.push(b)));
    allBd.sort((a, b) => a.t_start - b.t_start);
    btnFault.addEventListener("click", () => {{
      if (!allBd.length) return;
      currentSimTime = Math.max(0, allBd[0].t_start - 90);
      scrub.value = String(Math.floor(currentSimTime));
      playing = true;
      syncPlayButton();
    }});
  }}

  function updateComparisonDeltas() {{
    if (!CONFIG.is_comparison || scenes.length < 2) return;
    const L = scenes[0].liveSummary;
    const R = scenes[1].liveSummary;
    const dOre = R.ore - L.ore;
    const dWst = R.waste - L.waste;
    const oreEl = document.getElementById(UID + "_d_ore");
    const wstEl = document.getElementById(UID + "_d_waste");
    const redEl = document.getElementById(UID + "_d_redir");
    const qEl   = document.getElementById(UID + "_d_queue");
    if (oreEl) {{
      oreEl.textContent = "Δ Mineral (Der − Izq): " + (dOre >= 0 ? "+" : "") + dOre.toFixed(1) + " t";
      oreEl.style.color = dOre >= 0 ? "#4ade80" : "#f87171";
    }}
    if (wstEl) {{
      wstEl.textContent = "Δ Desmonte: " + (dWst >= 0 ? "+" : "") + dWst.toFixed(1) + " t";
    }}
    if (redEl) {{
      redEl.textContent = "Redirecciones: Izq " + L.redirs + " vs Der " + R.redirs;
    }}
    if (qEl) {{
      qEl.textContent = "En cola ahora: Izq " + L.activeQueues + " vs Der " + R.activeQueues;
    }}
  }}

  function tick(nowMs) {{
    const dtReal = Math.min(0.1, Math.max(0, (nowMs - lastFrameMs) / 1000));
    lastFrameMs = nowMs;

    if (playing) {{
      currentSimTime += dtReal * speed;
      if (currentSimTime >= maxHorizon) {{
        currentSimTime = maxHorizon;
        playing = false;
        syncPlayButton();
      }}
      scrub.value = String(Math.floor(currentSimTime));
    }}

    clockEl.textContent = formatHMS(currentSimTime) + " / " + formatHMS(maxHorizon);
    scenes.forEach(scene => renderScene(scene, currentSimTime, speed, nowMs));
    updateComparisonDeltas();
    requestAnimationFrame(tick);
  }}

  requestAnimationFrame(tick);
}})();
</script>
</body>
</html>
"""


def build_2d_viewer_html(trace_payload: Dict[str, Any], height: int = 640) -> str:
    """
    Genera el visor HTML5 Canvas interactivo para una simulación individual.

    Parámetros:
    - trace_payload: diccionario retornado por `SubHaulDESEngine.get_trace_payload()`.
    - height: altura total sugerida (px) del iframe en Streamlit.
    """
    return _render_2d_html([trace_payload], height=height, is_comparison=False)


def build_2d_comparison_html(
    trace_left: Dict[str, Any],
    trace_right: Dict[str, Any],
    height: int = 560,
) -> str:
    """
    Genera la vista comparativa lado a lado (2 canvas sincronizados con un solo
    juego de controles y barra de deltas en tiempo real).

    Parámetros:
    - trace_left: payload de la política izquierda (ej. FIJA).
    - trace_right: payload de la política derecha (ej. DIN_ETA_RITMO).
    - height: altura total sugerida (px) del iframe en Streamlit.
    """
    return _render_2d_html([trace_left, trace_right], height=height, is_comparison=True)


# ==============================================================================
# PRUEBA RÁPIDA STANDALONE (ABRE EN NAVEGADOR ANTES DE INTEGRAR EN app.py)
# ==============================================================================
if __name__ == "__main__":
    from motor_des import BreakdownConfig, SubHaulDESEngine, build_huancapeti_base_config

    cfg = build_huancapeti_base_config(
        n_her=6,
        n_car=2,
        n_cot=7,
        policy="DIN_ETA_RITMO",
        prob_mode="FORECAST",
        seed=25,
    )
    cfg.breakdowns.append(
        BreakdownConfig(
            event_id="FALLA_COT_180",
            station_key="COT",
            mode="SCHEDULED",
            start_min=180.0,
            duration_min=60.0,
            units_affected=1,
            active=True,
        )
    )

    eng = SubHaulDESEngine(cfg, record_trace=True)
    kpis = eng.run()
    payload = eng.get_trace_payload()

    out_file = Path("prueba_2d.html")
    out_file.write_text(build_2d_viewer_html(payload, height=680), encoding="utf-8")

    print("=" * 90)
    print(f"Visor 2D generado correctamente: {out_file.resolve()}")
    print(
        f"Escenario: 6-2-7 | {payload['policy_label']} | Falla COT min 180 (+60 min)\n"
        f"Mineral: {kpis['ore_total_tn']:.2f} t | Desmonte: {kpis['waste_total_tn']:.2f} t | "
        f"Ingreso Neto: ${kpis['net_revenue_usd']:,.2f}\n"
        f"Eventos -> travels: {len(payload['travels'])}, stations: {len(payload['stations'])}, "
        f"dumps: {len(payload['dumps'])}, breakdowns: {len(payload['breakdowns'])}, "
        f"redirections: {len(payload['redirections'])}"
    )
    print("=" * 90)
    print("Abre 'prueba_2d.html' en tu navegador y prueba el botón '🚨 Ir a falla'.")