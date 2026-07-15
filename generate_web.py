"""
Build the static site from the results/ tree.

    python generate_web.py            # -> docs/index.html

Reads only what already exists; never touches models or data. Re-run after a
train/evaluate/forecast pass, then `git add docs && git push` — that push IS the release,
since GitHub Pages serves docs/ straight off the branch.

The output is a single self-contained HTML file with the data inlined: no fetch, no CDN, no
relative paths, so it renders identically opened from disk and served from Pages.
"""

import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)

OUT = os.path.join('docs', 'index.html')
FORECAST_GLOB = 'results/*/evaluation/*/*/*_forecast.csv'

# Reading order, simplest first: trees, then plain sequence models, then the two that regress
# a residual against a baseline. It doubles as the colour assignment — a model's slot here is
# its hue everywhere, so it keeps that hue no matter which others are on screen or which zone
# is showing. Colour follows the entity, never its rank in a filtered list.
MODEL_ORDER = [
    'xgboost', 'lightgbm',
    'transformer', 'lstm', 'mstnn', 'moe_transformer',
    'xgboost_residual', 'transformer_residual',
]

# The dataviz reference palette, in its published slot order — that order IS the
# colourblind-safety mechanism (worst adjacent CVD ΔE 24.2 light / 10.3 dark), not decoration.
PALETTE_LIGHT = ['#2a78d6', '#1baf7a', '#eda100', '#008300',
                 '#4a3aa7', '#e34948', '#e87ba4', '#eb6834']
PALETTE_DARK  = ['#3987e5', '#199e70', '#c98500', '#008300',
                 '#9085e9', '#e66767', '#d55181', '#d95926']

ZONES = {
    'dom': {'label': 'DOM', 'name': 'Dominion Energy Virginia'},
    'bge': {'label': 'BGE', 'name': 'Baltimore Gas & Electric'},
}

# Matches src/prediction_engine.FORECAST_HORIZON_DAYS: the forecast reaches two days past the
# last complete day of data, and those trailing days are exactly the ones with no truth yet.
DAY_AHEAD_DAYS = 2


def metrics(actual, pred):
    """The scoreboard row for one model, over the whole scored window.

    Computed here rather than in the browser on purpose. BRS depends on pandas' qcut binning,
    and a hand-rolled equal-count split in JS lands up to 0.003 MW/MW away from it — enough to
    print a different number at four decimals. The site would then quietly disagree with the
    console for the same model on the same data. Same formulas as
    src/models/_eval_utils.EvalUtils.evaluate_one, so the two always read the same.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    resid = p - a

    # Binned residual slope: error regressed on load. Ten equal-frequency bins by actual load,
    # mean residual in each, OLS through those ten points. It catches the bias the mean error
    # hides — a model can average out to zero and still miss every peak.
    bins = pd.qcut(a, q=10, labels=False, duplicates='drop')
    centres = pd.Series(a).groupby(bins).mean().values
    resids  = pd.Series(resid).groupby(bins).mean().values
    brs = float(np.polyfit(centres, resids, 1)[0])
    me = float(resid.mean())

    return {
        'mape': round(mean_absolute_percentage_error(a, p) * 100, 2),
        'mae':  round(float(mean_absolute_error(a, p)), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(a, p))), 2),
        'me':   round(me, 2),
        'brs':  round(brs, 4),
        'hours': int(len(a)),
    }


def build_payload():
    zones = {}
    for path in sorted(glob.glob(FORECAST_GLOB)):
        parts = path.split(os.sep)
        ds, model = parts[1], parts[3]

        df = pd.read_csv(path)
        # A forecast written before the DST fix has a blind 24 rows per day and no UTC column:
        # a phantom 2 a.m. in March, a missing hour in November. It is stale, and plotting it
        # next to fresh ones would look completely normal. Name it and stop, rather than draw
        # a chart that is quietly wrong for one model.
        if 'datetime_utc' not in df.columns:
            raise SystemExit(
                f"{path} is from an older run (no datetime_utc column, 24 rows/day regardless "
                f"of DST).\nEither re-run the forecast for {ds}/{model} — enable it in "
                f"PREDICT_CONFIG and run Model_Prediction.py — or delete the file if that model "
                f"is no longer part of the site."
            )
        df['datetime']     = pd.to_datetime(df['datetime'])
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
        pred_col = next(c for c in df.columns if c.endswith('_pred'))
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        df['hour'] = df['datetime'].dt.hour

        z = zones.setdefault(ds, {
            'label': ZONES.get(ds, {}).get('label', ds.upper()),
            'name':  ZONES.get(ds, {}).get('name', ''),
            'models': [], 'hours': {}, 'series': {}, 'actual': {},
        })
        z['models'].append(model)

        for date, g in df.groupby('date'):
            # Sort by UTC, not by the local clock: on a fall-back day 01:00 appears twice and
            # the local timestamp cannot order those two rows. A day holds 23, 24 or 25 hours
            # and the page plots exactly that many points — no phantom 2 a.m. in March, no
            # missing hour in November.
            g = g.sort_values('datetime_utc')
            z['hours'].setdefault(date, [int(h) for h in g['hour']])
            z['series'].setdefault(date, {})[model] = [round(v, 1) for v in g[pred_col]]
            # Preliminary load is the near-real-time actual — but only where PJM publishes it
            # for THIS zone. For bge it is the MIDATL regional aggregate, so the column is
            # absent by design (PREDICT_CONFIG['compare_to_preliminary']) and there is simply
            # no actual to draw rather than a rescaled stand-in.
            if 'preliminary_load' in g.columns and g['preliminary_load'].notna().any():
                z['actual'][date] = [None if pd.isna(v) else round(v, 1)
                                     for v in g['preliminary_load']]

    for z in zones.values():
        z['models'] = [m for m in MODEL_ORDER if m in z['models']]
        z['dates'] = sorted(z['series'])
        # The trailing days are the genuine day-ahead ones: they reach past the end of the
        # data, so they have no actual by construction rather than by accident.
        z['dayAhead'] = z['dates'][-DAY_AHEAD_DAYS:]
        # The rest already happened and PJM has published preliminary load for them — these
        # are the days the forecast can actually be scored on, which is the real-time test.
        # The error CURVES are derived in the page (series - actual: a subtraction the browser
        # can do, and shipping both copies would let them drift). The scoreboard NUMBERS are
        # not: see metrics().
        z['scored'] = [d for d in z['dates'] if d in z['actual']]

        z['metrics'] = {}
        for m in z['models']:
            a, p = [], []
            for d in z['scored']:
                for act, pred in zip(z['actual'][d], z['series'][d][m]):
                    if act is not None and pred is not None:
                        a.append(act)
                        p.append(pred)
            if a:
                z['metrics'][m] = metrics(a, p)

    return {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'modelOrder': MODEL_ORDER,
        'paletteLight': PALETTE_LIGHT,
        'paletteDark': PALETTE_DARK,
        'zones': zones,
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PJM Zonal Load Forecast</title>
<style>
:root {
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10); --hover:rgba(11,11,11,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --hover:rgba(255,255,255,.06);
  }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--plane); color:var(--ink);
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1360px; margin:0 auto; padding:24px 22px 56px; }
header h1 { margin:0 0 4px; font-size:22px; font-weight:650; letter-spacing:-.01em; }
header p { margin:0; color:var(--muted); font-size:13px; }

.tabs { display:flex; gap:4px; margin-top:26px; border-bottom:1px solid var(--border); }
.tab { appearance:none; background:none; border:0; cursor:pointer; color:var(--ink-2);
       font:inherit; font-size:14px; padding:10px 16px; border-bottom:2px solid transparent;
       margin-bottom:-1px; }
.tab:hover { color:var(--ink); }
.tab[aria-selected="true"] { color:var(--ink); font-weight:600; border-bottom-color:var(--ink); }

/* Zone lives one level down, deliberately: it is not a peer of "which view am I in", it is
   "which system am I looking at". The two zones are separate 5x-apart load curves that must
   never share an axis, so the UI never lets them be on screen together. */
.subtabs { display:flex; gap:6px; margin-top:16px; }
.subtab { appearance:none; cursor:pointer; font:inherit; font-size:12px; padding:5px 13px;
          border-radius:999px; border:1px solid var(--border); background:var(--surface);
          color:var(--ink-2); }
.subtab:hover { background:var(--hover); color:var(--ink); }
.subtab[aria-selected="true"] { background:var(--ink); color:var(--surface);
                                border-color:var(--ink); font-weight:600; }
.subtab small { opacity:.65; margin-left:6px; font-size:11px; }

.panel { display:none; padding-top:18px; }
.panel[data-active] { display:block; }

.controls { display:flex; flex-wrap:wrap; gap:22px; align-items:flex-end; margin-bottom:18px; }
.field { display:flex; flex-direction:column; gap:6px; }
.field > label { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.seg { display:flex; border:1px solid var(--border); border-radius:7px; overflow:hidden; }
.seg button { appearance:none; border:0; background:var(--surface); color:var(--ink-2);
              font:inherit; font-size:13px; padding:7px 15px; cursor:pointer; white-space:nowrap; }
.seg button + button { border-left:1px solid var(--border); }
.seg button:hover { background:var(--hover); color:var(--ink); }
.seg button[aria-pressed="true"] { background:var(--ink); color:var(--surface); font-weight:600; }
.spacer { flex:1; }

.layout { display:grid; grid-template-columns:minmax(0,1fr) 224px; gap:20px; align-items:start; }
@media (max-width:820px) { .layout { grid-template-columns:1fr; } }
.stack { display:flex; flex-direction:column; gap:16px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; }
.card > h3 { margin:0 0 2px; font-size:13px; font-weight:600; }
.card > h3 + p { margin:0 0 10px; font-size:12px; color:var(--muted); }

.chart-wrap { position:relative; }
svg { display:block; width:100%; overflow:visible; }
.series { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
.gridline { stroke:var(--grid); stroke-width:1; }
.axisline { stroke:var(--axis); stroke-width:1; }
.zeroline { stroke:var(--ink-2); stroke-width:1.5; }
.tick { fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }
.axis-title { fill:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.crosshair { stroke:var(--axis); stroke-width:1; stroke-dasharray:3 3; }
.dot { stroke:var(--surface); stroke-width:2; }

.legend-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px; }
.legend-head span { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
.legend-head button { appearance:none; border:0; background:none; color:var(--ink-2);
                      cursor:pointer; font:inherit; font-size:11px; text-decoration:underline; padding:0; }
.legend-item { display:flex; align-items:center; gap:8px; padding:5px 6px; border-radius:6px;
               cursor:pointer; font-size:13px; color:var(--ink-2); user-select:none; }
.legend-item:hover { background:var(--hover); color:var(--ink); }
.legend-item.off { opacity:.4; }
.legend-item.emph { background:var(--hover); color:var(--ink); }
.legend-item .mape { margin-left:auto; font-size:11px; font-variant-numeric:tabular-nums;
                     color:var(--muted); }
.swatch { width:11px; height:11px; border-radius:3px; flex:none; }
.legend-item input { display:none; }
.legend-note { margin:10px 0 0; font-size:11px; color:var(--muted); line-height:1.45; }

.tooltip { position:absolute; pointer-events:none; opacity:0; transition:opacity .1s;
           background:var(--surface); border:1px solid var(--border); border-radius:8px;
           padding:9px 11px; font-size:12px; box-shadow:0 6px 20px rgba(0,0,0,.13);
           min-width:180px; z-index:5; }
.tooltip h4 { margin:0 0 6px; font-size:11px; color:var(--muted); font-weight:500;
              text-transform:uppercase; letter-spacing:.06em; }
.tt-row { display:flex; align-items:center; gap:7px; padding:2px 0; color:var(--ink-2); }
.tt-row.emph { color:var(--ink); font-weight:650; }
.tt-row .v { margin-left:auto; font-variant-numeric:tabular-nums; }

.note { margin:14px 0 0; font-size:12px; color:var(--muted); }
.note b { color:var(--ink-2); font-weight:600; }
.empty { padding:40px 16px; text-align:center; color:var(--muted); font-size:13px; line-height:1.6; }
table { border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }
th, td { padding:5px 9px; text-align:right; border-bottom:1px solid var(--border); }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:500; font-size:11px; text-transform:uppercase;
           letter-spacing:.05em; }
tbody tr:hover { background:var(--hover); }
td .dir { color:var(--muted); font-size:11px; }
.table-scroll { overflow-x:auto; }
.hidden { display:none; }

/* Scoreboard = master selector. Rows carry a checkbox, dim when off, and sort by any column. */
#r-metrics-table thead th { cursor:pointer; user-select:none; white-space:nowrap; }
#r-metrics-table thead th:hover { color:var(--ink); }
#r-metrics-table thead th[aria-sort] { color:var(--ink); font-weight:650; }
#r-metrics-table thead th[aria-sort]::after { content:" \25BC"; font-size:9px; }
#r-metrics-table thead th[aria-sort="ascending"]::after { content:" \25B2"; font-size:9px; }
/* No checkboxes: the row's own dimming IS the selected state. Click anywhere on a row. */
#r-metrics-table tbody tr { cursor:pointer; }
#r-metrics-table tbody tr.off { opacity:.35; }
#r-metrics-table tbody tr.emph { background:var(--hover); }
#r-metrics-table td:first-child { display:flex; align-items:center; gap:8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>PJM Zonal Load Forecast</h1>
    <p>Day-ahead 24-hour forecasts &middot; generated __GENERATED__</p>
  </header>

  <div class="tabs" role="tablist">
    <button class="tab" role="tab" id="tab-dayahead" aria-selected="true">Day-Ahead Prediction</button>
    <button class="tab" role="tab" id="tab-realtime" aria-selected="false">Real-Time Test</button>
  </div>
  <div class="subtabs" role="tablist" id="zone-tabs"></div>

  <!-- ======================= Tab 1: day-ahead ======================= -->
  <section class="panel" id="panel-dayahead" data-active>
    <div class="controls">
      <div class="field"><label>Forecast day</label><div class="seg" id="d-date"></div></div>
      <div class="spacer"></div>
      <div class="field"><label>View</label><div class="seg" id="d-view"></div></div>
    </div>
    <div class="layout">
      <div class="card">
        <div class="chart-wrap" id="d-chartwrap">
          <svg id="d-chart" height="480"></svg>
          <div class="tooltip" id="d-tip"></div>
        </div>
        <div class="table-scroll hidden" id="d-table"></div>
        <p class="note" id="d-note"></p>
      </div>
      <div class="card">
        <div class="legend-head"><span>Models</span><button id="d-all">toggle all</button></div>
        <div id="d-legend"></div>
      </div>
    </div>
  </section>

  <!-- ======================= Tab 2: real-time test ======================= -->
  <section class="panel" id="panel-realtime">
    <!-- Scoreboard first: it is the summary the whole tab exists to deliver, and it doubles
         as the master model selector — the charts below follow whatever is ticked here. -->
    <div class="card" id="r-metrics">
      <div class="legend-head" style="margin-bottom:4px">
        <h3 style="margin:0">Scoreboard</h3>
        <button id="r-all">toggle all</button>
      </div>
      <p id="r-metrics-sub"></p>
      <div class="table-scroll"><div id="r-metrics-table"></div></div>
      <p class="note">
        Tick a model to show it below; click a column to sort. <b>ME</b> mean error &mdash; the
        constant part of the bias (over / under). <b>BRS</b> binned residual slope &mdash; error
        regressed on load, catching the bias the ME hides: a model can average to zero and still
        miss every peak. ME and BRS sort by magnitude (nearest zero first).
      </p>
    </div>

    <div class="controls" id="r-controls" style="margin-top:18px">
      <div class="field"><label>Day</label><div class="seg" id="r-date"></div></div>
    </div>
    <div class="layout">
      <div class="stack" id="r-charts">
        <div class="card">
          <h3>Forecast vs actual</h3>
          <p>Preliminary load is the baseline &mdash; PJM's near-real-time published load for this zone.</p>
          <div class="chart-wrap">
            <svg id="r-load" height="400"></svg>
            <div class="tooltip" id="r-load-tip"></div>
          </div>
        </div>
        <div class="card">
          <h3>Error</h3>
          <p>Forecast minus actual. Above zero = over-forecast, below = under.</p>
          <div class="chart-wrap">
            <svg id="r-err" height="270"></svg>
            <div class="tooltip" id="r-err-tip"></div>
          </div>
        </div>
      </div>
      <div class="card" id="r-side">
        <div class="legend-head"><span>Models &middot; this day</span><button id="r-legend-all">toggle all</button></div>
        <div id="r-legend"></div>
        <p class="legend-note" id="r-legend-note"></p>
      </div>
    </div>

    <div class="card hidden" id="r-empty">
      <p class="empty" id="r-empty-text"></p>
    </div>
  </section>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const $ = id => document.getElementById(id);
const ACTUAL = '__actual__';

const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches;
// Colour is keyed off the model's slot in the FIXED order, never its index in the visible
// list — so hiding one model can never repaint the ones still on screen.
const colourOf = m => (isDark() ? DATA.paletteDark : DATA.paletteLight)[DATA.modelOrder.indexOf(m)];
const ink = () => getComputedStyle(document.body).getPropertyValue('--ink').trim();
const pretty = m => m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
/* en-US, not the viewer's locale: a German browser groups 15,425 as "15.425" and an Indian one
   as "15,425" with different grouping again. A report page must not change its numbers
   depending on who opens it. */
const num = v => Math.round(v).toLocaleString('en-US');
const fmt = v => v == null ? '—' : num(v);

/* Zone is global: one sub-tab under the main tabs, shared by both views. Both panels read it,
   so switching zone keeps you where you are instead of resetting the view. */
let ZONE = Object.keys(DATA.zones)[0];

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.setAttribute('aria-selected', x === t));
  document.querySelectorAll('.panel').forEach(p =>
    p.toggleAttribute('data-active', p.id === 'panel-' + t.id.slice(4)));
  renderDay(); renderRT();      // sizes are only measurable once a panel is actually visible
});

function renderZoneTabs() {
  const el = $('zone-tabs');
  el.innerHTML = '';
  for (const k of Object.keys(DATA.zones)) {
    const z = DATA.zones[k];
    const b = document.createElement('button');
    b.className = 'subtab';
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', ZONE === k);
    b.innerHTML = `${z.label}<small>${z.name}</small>`;
    b.onclick = () => {
      ZONE = k;
      D.date = R.date = null;       // day lists differ per zone
      D.off.clear(); R.off.clear(); // and so do the model lists
      renderZoneTabs(); renderDay(); renderRT();
    };
    el.appendChild(b);
  }
}

function seg(el, items, current, pick) {
  el.innerHTML = '';
  for (const [value, label] of items) {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', current === value);
    b.onclick = () => pick(value);
    el.appendChild(b);
  }
}

/* Pinned to en-US, not the viewer's locale. This is a report page: it has to read the same
   for whoever opens it, and a browser set to another language would otherwise render the
   dates in that language inside an otherwise English page. 'T00:00' (not a bare date) parses
   as LOCAL midnight — a bare '2026-07-13' is parsed as UTC and shows as the 12th west of
   Greenwich. */
const dayLabel = d => new Date(d + 'T00:00').toLocaleDateString('en-US',
  { weekday: 'short', month: 'short', day: 'numeric' });

/* ============================================================================
   Chart — one renderer, used by all three plots.
   `zero: true` pins the y-scale around 0 and draws the zero rule, which is what turns the
   same code into an error chart. `onEmph` lets a hovered line light up in BOTH charts of the
   real-time tab at once, so the load curve and its error stay visually tied together.
   ========================================================================== */
function makeChart(svg, tip, { zero = false, onEmph = null } = {}) {
  const M = { top: 16, right: 16, bottom: 34, left: 60 };
  let geom = null, getSeries = () => [], getEmph = () => null, getTitle = () => '',
      getHours = () => [];

  function paint() {
    const series = getSeries(), emph = getEmph(), HRS = getHours(), N = HRS.length;
    const W = svg.clientWidth || 700, H = svg.height.baseVal.value;
    const iw = W - M.left - M.right, ih = H - M.top - M.bottom;
    const vals = series.flatMap(s => s.values).filter(v => v != null);

    if (!vals.length) {
      svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" class="tick">No series selected</text>`;
      geom = null; return;
    }

    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (zero) { const m = Math.max(Math.abs(lo), Math.abs(hi)) * 1.15 || 1; lo = -m; hi = m; }
    else { const p = (hi - lo) * 0.12 || 1; lo -= p; hi += p; }

    // Indexed by POSITION in the day, not by clock hour: a fall-back day has two 01:00s and
    // they must sit at two different x positions.
    const x = i => M.left + iw * i / Math.max(1, N - 1);
    const y = v => M.top + ih - ih * (v - lo) / (hi - lo);
    geom = { x, y, iw, ih, series, M, HRS, N };

    const step = Math.pow(10, Math.floor(Math.log10(hi - lo))) / 2;
    const p = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
      p.push(`<line class="gridline" x1="${M.left}" x2="${M.left + iw}" y1="${y(t)}" y2="${y(t)}"/>`,
             `<text class="tick" x="${M.left - 9}" y="${y(t) + 4}" text-anchor="end">${num(t)}</text>`);
    }
    if (zero) p.push(`<line class="zeroline" x1="${M.left}" x2="${M.left + iw}" y1="${y(0)}" y2="${y(0)}"/>`);
    else p.push(`<line class="axisline" x1="${M.left}" x2="${M.left + iw}" y1="${M.top + ih}" y2="${M.top + ih}"/>`);

    const every = N > 26 ? 3 : 3;
    for (let i = 0; i < N; i++) {
      if (i % every && i !== N - 1) continue;
      p.push(`<text class="tick" x="${x(i)}" y="${M.top + ih + 18}" text-anchor="middle">${String(HRS[i]).padStart(2, '0')}</text>`);
    }
    p.push(`<text class="axis-title" x="${M.left}" y="${M.top - 3}">${getTitle()}</text>`,
           `<text class="axis-title" x="${M.left + iw}" y="${M.top + ih + 32}" text-anchor="end">Hour (EPT)</text>`);

    /* Draw order and opacity are the whole legibility story here.
       - The baseline (`top`) is painted LAST, so it sits above the model lines instead of
         under them, and it NEVER fades: it is the thing everything else is judged against, so
         dimming it while you inspect a model hides exactly the curve you are comparing to.
         The surface-coloured casing under it is what keeps one dark line readable while it
         crosses a bundle of bright ones — thickness alone does not do it.
       - Deselected models (`faint`) stay on the canvas as ghosts rather than vanishing, so
         unchecking one still leaves you its shape for context. They are not hover targets and
         they are not in the tooltip; they are background, not data you are reading. */
    const opacity = s => s.top ? 1
                       : s.faint ? .17
                       : (emph && emph !== s.key) ? .16
                       : 1;
    const order = s => (s.top ? 2 : s.faint ? 0 : 1);
    for (const s of [...series].sort((a, b) => order(a) - order(b))) {
      const pts = s.values.map((v, i) => v == null ? null : `${x(i)},${y(v)}`).filter(Boolean).join(' ');
      const w = emph === s.key ? s.width + 1.5 : s.width;
      if (s.top) {
        p.push(`<polyline class="series" points="${pts}" stroke="var(--surface)" ` +
               `stroke-width="${w + 4}" opacity=".92"/>`);
      }
      p.push(`<polyline class="series" points="${pts}" stroke="${s.colour}" ` +
             `opacity="${opacity(s)}" stroke-width="${s.faint ? 1.5 : w}"/>`);
    }
    svg.innerHTML = p.join('');
  }

  /* Hover does both jobs at once: the crosshair reads EVERY visible series at that hour, and
     the line nearest the cursor is named and emphasised — so pointing at a line tells you
     which model it is, and pointing at an hour tells you what all of them said. */
  svg.addEventListener('mousemove', ev => {
    if (!geom) return;
    const r = svg.getBoundingClientRect();
    const px = ev.clientX - r.left, py = ev.clientY - r.top;
    if (px < M.left - 10 || px > M.left + geom.iw + 10) return hide();

    const i = Math.max(0, Math.min(geom.N - 1,
      Math.round((px - M.left) / geom.iw * Math.max(1, geom.N - 1))));
    // Ghosts are background, not readable data: they cannot be picked and they are not listed.
    const live = geom.series.filter(s => !s.faint);

    let near = null, best = Infinity;
    for (const s of live) {
      const v = s.values[i];
      if (v == null) continue;
      const d = Math.abs(geom.y(v) - py);
      if (d < best) { best = d; near = s.key; }
    }
    const emph = best < 28 ? near : null;
    if (onEmph) onEmph(emph); else { getEmph = () => emph; paint(); }

    const layer = [`<line class="crosshair" x1="${geom.x(i)}" x2="${geom.x(i)}" y1="${M.top}" y2="${M.top + geom.ih}"/>`];
    for (const s of [...live].sort((a, b) => (a.top ? 1 : 0) - (b.top ? 1 : 0))) {
      const v = s.values[i];
      if (v == null) continue;
      const on = getEmph() === s.key;
      // The baseline's marker never fades either — same reason its line does not.
      const faded = getEmph() && !on && !s.top;
      layer.push(`<circle class="dot" cx="${geom.x(i)}" cy="${geom.y(v)}" r="${on ? 6 : (s.top ? 5.5 : 4)}" ` +
                 `fill="${s.colour}" opacity="${faded ? .22 : 1}"/>`);
    }
    svg.insertAdjacentHTML('beforeend', layer.join(''));

    tip.innerHTML = `<h4>${hourLabel(geom.HRS, i)} EPT &middot; ${getTitle()}</h4>` +
      live.filter(s => s.values[i] != null)
        .sort((a, b) => b.values[i] - a.values[i])
        .map(s => `<div class="tt-row${getEmph() === s.key ? ' emph' : ''}">` +
                  `<span class="swatch" style="background:${s.colour}"></span><span>${s.name}</span>` +
                  `<span class="v">${s.values[i] > 0 && zero ? '+' : ''}${fmt(s.values[i])}</span></div>`).join('');
    tip.style.opacity = 1;
    tip.style.left = Math.min(geom.x(h) + 14, r.width - tip.offsetWidth - 4) + 'px';
    tip.style.top = Math.max(4, py - 12) + 'px';
  });

  function hide() { tip.style.opacity = 0; if (onEmph) onEmph(null); else { getEmph = () => null; paint(); } }
  svg.addEventListener('mouseleave', hide);

  return {
    render(series, emphGetter, title, hours) {
      getSeries = () => series;
      getEmph = emphGetter;
      getTitle = () => title;
      getHours = () => hours;
      paint();
    },
    repaint: paint,
  };
}

/* On a fall-back day 01:00 happens twice. Both are real hours with different real loads, so
   both are plotted — and the second is marked so the repeat reads as the DST hour it is,
   rather than as a rendering glitch. */
function hourLabel(hours, i) {
  const h = String(hours[i]).padStart(2, '0') + ':00';
  return (i > 0 && hours[i] === hours[i - 1]) ? h + ' (2nd)' : h;
}

/* ============================================================================
   Legend — also the multi-select. One legend can drive several charts.
   ========================================================================== */
function renderLegend(el, keys, state, onChange, mape) {
  el.innerHTML = '';
  for (const k of keys) {
    const actual = k === ACTUAL;
    const row = document.createElement('label');
    row.className = 'legend-item' + (state.off.has(k) ? ' off' : '');
    row.innerHTML =
      `<input type="checkbox"${state.off.has(k) ? '' : ' checked'}>` +
      `<span class="swatch" style="background:${actual ? ink() : colourOf(k)}"></span>` +
      `<span>${actual ? 'Actual (prelim.)' : pretty(k)}</span>` +
      (mape && mape[k] != null ? `<span class="mape">${mape[k].toFixed(2)}%</span>` : '');
    row.querySelector('input').onchange = e => {
      e.target.checked ? state.off.delete(k) : state.off.add(k);
      onChange();
    };
    row.onmouseenter = () => { state.emph = k; onChange(true); };
    row.onmouseleave = () => { state.emph = null; onChange(true); };
    el.appendChild(row);
  }
}

/* ============================================================================
   Tab 1 — day-ahead
   ========================================================================== */
const D = { date: null, view: 'chart', off: new Set(), emph: null };
const dChart = makeChart($('d-chart'), $('d-tip'), { onEmph: e => { D.emph = e; dChart.repaint(); } });

function dSeries() {
  const z = DATA.zones[ZONE], out = [];
  if (z.actual[D.date] && !D.off.has(ACTUAL))
    out.push({ key: ACTUAL, name: 'Actual (prelim.)', values: z.actual[D.date],
               colour: ink(), width: 3.5, top: true });
  for (const m of z.models) {
    out.push({ key: m, name: pretty(m), values: z.series[D.date][m], colour: colourOf(m),
               width: 2, faint: D.off.has(m) });
  }
  return out;
}

$('d-all').onclick = () => {
  const z = DATA.zones[ZONE];
  const all = [...z.models, ...(z.actual[D.date] ? [ACTUAL] : [])];
  if (D.off.size) D.off.clear(); else all.forEach(k => D.off.add(k));
  renderDay();
};

function renderDay(paintOnly) {
  const z = DATA.zones[ZONE];
  if (!z.dayAhead.includes(D.date)) D.date = z.dayAhead[z.dayAhead.length - 1];

  if (paintOnly) { dChart.repaint(); return; }

  seg($('d-date'), z.dayAhead.map(d => [d, dayLabel(d)]), D.date, v => { D.date = v; renderDay(); });
  seg($('d-view'), [['chart', 'Chart'], ['table', 'Table']], D.view, v => { D.view = v; renderDay(); });

  const keys = [...(z.actual[D.date] ? [ACTUAL] : []), ...z.models];
  renderLegend($('d-legend'), keys, D, renderDay);

  const chart = D.view === 'chart';
  $('d-chartwrap').classList.toggle('hidden', !chart);
  $('d-table').classList.toggle('hidden', chart);
  const hrs = z.hours[D.date];
  if (chart) dChart.render(dSeries(), () => D.emph, 'MW', hrs);
  else $('d-table').innerHTML = table(dSeries(), hrs);

  $('d-note').innerHTML = z.actual[D.date]
    ? `<b>${z.label}</b> &middot; ${z.name}. Actual is PJM's preliminary load for this zone.`
    : `<b>${z.label}</b> &middot; ${z.name}. No actual line: this day has not happened yet, so PJM has published nothing to compare against.`;
}

/* ============================================================================
   Tab 2 — real-time test: the days that HAVE a published actual
   ========================================================================== */
// off / emph are the tab's ONE shared selection+hover state. Scoreboard, side legend and both
// charts all read it and all write it, so ticking a model anywhere updates everywhere at once.
// sort is the scoreboard's column; ME and BRS sort by magnitude (nearest zero first).
const R = { date: null, off: new Set(), emph: null, sort: { key: 'mape', dir: 1 } };

const SCORE_COLS = [
  { key: 'model', label: 'Model' },
  { key: 'mape',  label: 'MAPE' },
  { key: 'mae',   label: 'MAE (MW)' },
  { key: 'rmse',  label: 'RMSE (MW)' },
  { key: 'me',    label: 'ME (MW)' },
  { key: 'brs',   label: 'BRS (MW/MW)' },
];
const sortVal = (r, key) =>
  key === 'model' ? DATA.modelOrder.indexOf(r.model)
  : (key === 'me' || key === 'brs') ? Math.abs(r[key])   // bias magnitude, either direction
  : r[key];
const rEmph = e => { R.emph = e; markEmph(); };   // hovering a chart line also lights its rows
const rLoad = makeChart($('r-load'), $('r-load-tip'), { onEmph: rEmph });
const rErr  = makeChart($('r-err'),  $('r-err-tip'),  { zero: true, onEmph: rEmph });

function rLoadSeries() {
  const z = DATA.zones[ZONE], out = [];
  if (!R.off.has(ACTUAL))
    out.push({ key: ACTUAL, name: 'Actual (prelim.)', values: z.actual[R.date],
               colour: ink(), width: 3.5, top: true });
  for (const m of z.models) {
    out.push({ key: m, name: pretty(m), values: z.series[R.date][m], colour: colourOf(m),
               width: 2, faint: R.off.has(m) });
  }
  return out;
}

/* Errors are derived here rather than shipped: the payload already holds both curves, and two
   copies of the same fact are two chances for them to disagree. */
function rErrSeries() {
  const z = DATA.zones[ZONE], a = z.actual[R.date], out = [];
  for (const m of z.models) {
    const p = z.series[R.date][m];
    out.push({ key: m, name: pretty(m), colour: colourOf(m), width: 2, faint: R.off.has(m),
               values: p.map((v, i) => a[i] == null ? null : v - a[i]) });
  }
  return out;
}

function rMape() {
  const z = DATA.zones[ZONE], a = z.actual[R.date], out = {};
  for (const m of z.models) {
    const p = z.series[R.date][m];
    const e = p.map((v, i) => a[i] == null ? null : Math.abs(v - a[i]) / a[i] * 100).filter(v => v != null);
    if (e.length) out[m] = e.reduce((x, y) => x + y, 0) / e.length;
  }
  return out;
}

/* ---------- scoreboard: master selector + sortable table ----------
   Numbers arrive precomputed (BRS leans on pandas' qcut binning; a hand-rolled split here
   lands up to 0.003 MW/MW away — enough to print a different figure at four decimals and have
   the page quietly contradict the console). The browser only sorts, filters and highlights.  */
function renderScoreboard() {
  const z = DATA.zones[ZONE];
  const rows = z.models.filter(m => z.metrics[m]).map(m => ({ model: m, ...z.metrics[m] }));
  rows.sort((a, b) => (sortVal(a, R.sort.key) - sortVal(b, R.sort.key)) * R.sort.dir);

  const head = SCORE_COLS.map(c =>
    `<th data-k="${c.key}"${R.sort.key === c.key
      ? ` aria-sort="${R.sort.dir > 0 ? 'ascending' : 'descending'}"` : ''}>${c.label}</th>`).join('');
  const body = rows.map(r => {
    const off = R.off.has(r.model);
    return `<tr data-m="${r.model}" class="${off ? 'off' : ''}${R.emph === r.model ? ' emph' : ''}">
      <td><span class="swatch" style="background:${colourOf(r.model)}"></span>${pretty(r.model)}</td>
      <td>${r.mape.toFixed(2)}%</td>
      <td>${r.mae.toFixed(2)}</td>
      <td>${r.rmse.toFixed(2)}</td>
      <td>${r.me >= 0 ? '+' : ''}${r.me.toFixed(2)} <span class="dir">(${r.me >= 0 ? 'over' : 'under'})</span></td>
      <td>${r.brs >= 0 ? '+' : ''}${r.brs.toFixed(4)} <span class="dir">(${r.brs >= 0 ? 'over&#8593; at peak' : 'under&#8595; at peak'})</span></td>
    </tr>`;
  }).join('');

  const el = $('r-metrics-table');
  el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  el.querySelectorAll('th').forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (R.sort.key === k) R.sort.dir *= -1; else R.sort = { key: k, dir: 1 };
    renderScoreboard();          // only the row order changes — no need to touch the charts
  });
  el.querySelectorAll('tbody tr').forEach(tr => {
    const m = tr.dataset.m;
    tr.onclick = () => { R.off.has(m) ? R.off.delete(m) : R.off.add(m); renderRT(); };
    tr.onmouseenter = () => { R.emph = m; markEmph(); };
    tr.onmouseleave = () => { R.emph = null; markEmph(); };
  });
}

/* Side legend: the same models, but ranked by THIS day's MAPE (so the order shifts day to
   day), and the same shared selection — ticking here moves the scoreboard and charts too. */
function renderRLegend() {
  const z = DATA.zones[ZONE], mape = rMape();
  const models = [...z.models].sort((a, b) => (mape[a] ?? Infinity) - (mape[b] ?? Infinity));
  const el = $('r-legend');
  el.innerHTML = '';
  for (const k of [ACTUAL, ...models]) {
    const actual = k === ACTUAL, off = R.off.has(k);
    const row = document.createElement('label');
    row.className = 'legend-item' + (off ? ' off' : '') + (R.emph === k ? ' emph' : '');
    row.dataset.k = k;
    row.innerHTML =
      `<input type="checkbox"${off ? '' : ' checked'}>` +
      `<span class="swatch" style="background:${actual ? ink() : colourOf(k)}"></span>` +
      `<span>${actual ? 'Actual (prelim.)' : pretty(k)}</span>` +
      (!actual && mape[k] != null ? `<span class="mape">${mape[k].toFixed(2)}%</span>` : '');
    row.querySelector('input').onchange = () => { off ? R.off.delete(k) : R.off.add(k); renderRT(); };
    row.onmouseenter = () => { R.emph = k; markEmph(); };
    row.onmouseleave = () => { R.emph = null; markEmph(); };
    el.appendChild(row);
  }
}

/* Hover is shared too: emphasising a model lights its line in both charts and its row in both
   the scoreboard and the legend. Repaint the charts, restyle the rows — no full rebuild. */
function markEmph() {
  rLoad.repaint(); rErr.repaint();
  document.querySelectorAll('#r-metrics-table tbody tr').forEach(tr =>
    tr.classList.toggle('emph', tr.dataset.m === R.emph));
  document.querySelectorAll('#r-legend .legend-item').forEach(li =>
    li.classList.toggle('emph', li.dataset.k === R.emph));
}

// Both toggle-all buttons write the same shared state. The scoreboard's covers the models it
// lists; the legend's also covers the actual baseline it carries, so its "all" means all.
function toggleAll(keys) {
  if (keys.every(k => R.off.has(k))) keys.forEach(k => R.off.delete(k));
  else keys.forEach(k => R.off.add(k));
  renderRT();
}
$('r-all').onclick        = () => toggleAll(DATA.zones[ZONE].models);
$('r-legend-all').onclick = () => toggleAll([ACTUAL, ...DATA.zones[ZONE].models]);

function renderRT(paintOnly) {
  const z = DATA.zones[ZONE];
  // A zone can only be scored where PJM publishes preliminary load for that zone itself. For
  // BGE it publishes the MIDATL regional aggregate — about 8.8x BGE's own load, and the ratio
  // drifts with the season. Dividing it down would manufacture a "truth" noisier than the
  // forecast it is supposed to judge, so this tab says so instead of drawing it.
  const empty = !z.scored.length;
  $('r-controls').classList.toggle('hidden', empty);
  $('r-charts').classList.toggle('hidden', empty);
  $('r-side').classList.toggle('hidden', empty);
  $('r-metrics').classList.toggle('hidden', empty);
  $('r-empty').classList.toggle('hidden', !empty);
  if (empty) {
    $('r-empty-text').innerHTML =
      `<b>${z.label}</b> has no actual to test against.<br>` +
      `PJM publishes no zone-level preliminary load for ${z.label} &mdash; only the MIDATL ` +
      `regional total, roughly 8.8&times; this zone's load. Rescaling it would invent a ` +
      `baseline noisier than the forecasts it is meant to judge, so nothing is scored here. ` +
      `Its true curve arrives with the verified metered load, about a week behind.<br><br>` +
      `Forecasts for ${z.label} are on the <b>Day-Ahead Prediction</b> tab.`;
    return;
  }

  if (!z.scored.includes(R.date)) R.date = z.scored[z.scored.length - 1];
  if (paintOnly) { rLoad.repaint(); rErr.repaint(); return; }

  const s = z.scored;
  const H = s.reduce((n, d) => n + z.hours[d].length, 0);   // real hours, DST-aware
  $('r-metrics-sub').textContent =
    `Whole test window: ${s.length} day${s.length > 1 ? 's' : ''} (${s[0]} → ${s[s.length - 1]}), ` +
    `${H} hours, vs PJM's preliminary load. Click a column to sort.`;
  renderScoreboard();

  seg($('r-date'), s.map(d => [d, dayLabel(d)]), R.date, v => { R.date = v; renderRT(); });
  renderRLegend();
  $('r-legend-note').textContent = "Ranked by this day's MAPE. Selection is shared with the scoreboard.";

  const hrs = z.hours[R.date];
  rLoad.render(rLoadSeries(), () => R.emph, 'MW', hrs);
  rErr.render(rErrSeries(), () => R.emph, 'MW error', hrs);
}

/* The table is not a nicety: three light-mode hues sit under 3:1 contrast, so the exact
   numbers have to be reachable without depending on colour at all. */
function table(all, hours) {
  // Ghosts are on the chart for shape, not for reading — a faint column of numbers would be
  // neither visible nor meaningfully excluded.
  const series = all.filter(s => !s.faint);
  if (!series.length) return '<p class="note">No series selected.</p>';
  // As many rows as the day has hours: 23 in March, 25 in November.
  return `<table><thead><tr><th>Hour (EPT)</th>${series.map(s => `<th>${s.name}</th>`).join('')}</tr></thead><tbody>` +
    hours.map((_, i) =>
      `<tr><td>${hourLabel(hours, i)}</td>` +
      series.map(s => `<td>${fmt(s.values[i])}</td>`).join('') + '</tr>').join('') +
    '</tbody></table>';
}

addEventListener('resize', () => { renderDay(true); renderRT(true); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',
  () => { renderZoneTabs(); renderDay(); renderRT(); });
renderZoneTabs();
renderDay();
renderRT();
</script>
</body>
</html>
"""



def main():
    payload = build_payload()
    if not payload['zones']:
        raise SystemExit(
            f"No files match {FORECAST_GLOB}. Run Model_Training.py — it forecasts as part of "
            f"the same pass — before generating the site."
        )

    for ds, z in payload['zones'].items():
        with_actual = sum(1 for d in z['dates'] if d in z['actual'])
        print(f"  {ds}: {len(z['models'])} models, {len(z['dates'])} days "
              f"({with_actual} with an actual) | day-ahead: {', '.join(z['dayAhead'])}")

    html = (HTML.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':')))
                .replace('__GENERATED__', payload['generated']))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write(html)
    print(f"\nWrote {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB, self-contained)")


if __name__ == '__main__':
    main()
