#!/usr/bin/env python3

# TF02-Pro LIDAR event logger
# Detects objects crossing the beam and saves per-event CSVs.
# J.Beale  5-Jun-2026

import serial
import numpy as np
import time
from datetime import datetime
import datetime as _dt
import pytz
import collections
import os
import threading
import json
import serial.tools.list_ports   # enable port scan for correct device
from flask import Flask, Response, render_template_string


# --- Configuration ---
VERSION = "1.035  2026-06-08"  # fix click by not rebuilding table on every SSE tick
OUTPUT_DIR    = 'events'

WEB_PORT      = 8080         # LAN web interface port
WEB_HZ        = 2            # SSE update rate for browser clients
WEB_BUF_SECS  = 50           # seconds of history kept for web display (10s shown)

SAMPLE_RATE   = 100          # nominal samples/sec
BASELINE_SECS = 4            # seconds of quiet data to seed the baseline
TRIGGER_N     = 3            # consecutive samples below threshold to trigger

TRIGGER_SIGMA = 6.0          # sigma below baseline mean to trigger
TRIGGER_OFFSET = 100         # additional fixed offset (counts) for trigger threshold

REARM_DIFFERENCE = 500       # absolute difference (mm) to re-arm
REARM_SECS    = 0.6          # seconds within baseline to consider event over
MAX_EVENT_SECS = 3.0         # hard cap on event recording length

PRE_TRIGGER   = 10           # samples to prepend from rolling buffer
EMA_ALPHA     = 0.001        # EMA coefficient for slow baseline drift (~1000-sample TC)

# Rise-time ratio: (lead + trail transition time) / total event duration, measured
# at 10%–90% of max excursion.  Pedestrians ~0.7–0.8; vehicles ~0.05.
RISE_TIME_THRESHOLD = 0.35  # >= this → pedestrian, < this → vehicle
DROPOUT_VALUES = {0, 45000} # mm: 0=overload, 45000=loss-of-signal
STABLE_RATE_THRESHOLD = 100 # mm/sample: max backward diff for a point to be "stable"

def find_ch341_port():
    """Return the serial port path for the first CH341 USB-serial adapter found."""
    matches = [
        p.device
        for p in serial.tools.list_ports.comports()
        if p.vid == 0x1a86 and p.pid == 0x7523
    ]
    if not matches:
        raise RuntimeError("CH341 USB-serial adapter not found")
    if len(matches) > 1:
        print(f"Warning: multiple CH341 devices found, using first: {matches}")
    return matches[0]


# Quiet-check threshold: only update baseline when per-batch std is low.
# From data, quiet individual-sample std ≈ 1–2 cm.  We gate on a live
# estimate, but seed it from the initial baseline period.
QUIET_STD_MULT = 4.0         # baseline update gated to: std < QUIET_STD_MULT * bg_std

# ---------------------------------------------------------------------------
# Shared state for web interface (written by main loop, read by Flask thread)
# ---------------------------------------------------------------------------
_web_lock   = threading.Lock()
_web_buf    = collections.deque()      # (epoch_float, dist_mm, strength, in_event)
_web_state  = {                        # latest single-sample snapshot
    "dist": 0, "dist_avg": 0, "strength": 0, "temp_c": 0.0,
    "bg_mean": 0.0, "bg_std": 0.0, "trigger": 0.0,
    "in_event": False, "category": "", "ts": 0.0,
    "last_event_id": 0.0,              # epoch of most recent completed event
}
# Most recent completed event — stored once, fetched on demand by browser
_last_event    = None   # dict: {id, category, bg_mean, samples: [[offset_ms, dist], ...]}
_recent_events = collections.deque(maxlen=10) # last 10 full events for selection/overlay
_event_log  = collections.deque(maxlen=10)  # last 10 event summaries for SSE
# All event epochs for the current calendar day — used for hourly/daily counts.
# A deque with a generous cap; at one event/sec for 24 h that's 86400 entries.
_event_epochs = collections.deque(maxlen=100000)

def _web_push(t, dist, strength, temp_c, in_ev):
    """Called from main loop after every sample read."""
    with _web_lock:
        _web_buf.append((t, dist, strength, in_ev))
        # Prune old samples
        cutoff = t - WEB_BUF_SECS
        while _web_buf and _web_buf[0][0] < cutoff:
            _web_buf.popleft()
        _web_state.update({
            "dist": dist, "strength": strength, "temp_c": temp_c,
            "bg_mean": bg_mean if 'bg_mean' in globals() else 0.0,
            "bg_std":  bg_std  if 'bg_std'  in globals() else 0.0,
            "trigger": (bg_mean - TRIGGER_SIGMA * bg_std - TRIGGER_OFFSET)
                       if 'bg_mean' in globals() else 0.0,
            "in_event": in_ev, "ts": t,
        })

# ---------------------------------------------------------------------------
# Flask web server  (runs in a daemon thread)
# ---------------------------------------------------------------------------
_flask_app = Flask(__name__)

_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LiDAR Monitor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');
  :root {
    --bg:    #f0f2f4;
    --panel: #ffffff;
    --bord:  #c8d0d8;
    --accent:#0068a8;
    --warn:  #cc2200;
    --text:  #1a2530;
    --dim:   #5a7080;
    --grid:  #d4dce4;
    --mono:  'Share Tech Mono', monospace;
    --head:  'Orbitron', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono);
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; padding: .6rem 1rem 2rem; }

  /* ── header row ── */
  #header { width: 100%; max-width: 760px; display: flex;
            align-items: baseline; justify-content: space-between;
            margin-bottom: .5rem; }
  #title  { font-family: var(--head); font-size: 1.05rem; letter-spacing: .22em;
            color: var(--accent); text-transform: uppercase; }
  #clock  { font-family: var(--mono); font-size: 1.05rem; color: var(--dim);
            letter-spacing: .08em; }
  #dot    { display: inline-block; width: .5rem; height: .5rem;
            border-radius: 50%; background: var(--bord); margin-right: .4rem;
            vertical-align: middle; }
  #dot.live { animation: blink1hz 1s step-start infinite; }
  @keyframes blink1hz {
    0%, 49% { background: var(--accent); box-shadow: 0 0 5px var(--accent); }
    50%,100% { background: var(--bord); box-shadow: none; }
  }

  /* ── compact distance readout bar ── */
  #readout-bar { display: flex; align-items: baseline; gap: .3rem;
                 margin-top: .4rem; margin-bottom: 0; flex-wrap: wrap; }
  .rdlabel { font-size: .78rem; color: var(--dim); }
  .rdval   { font-family: var(--mono); font-size: 1.15rem; color: var(--accent);
             font-weight: bold; min-width: 5em; }
  .rdval.ev { color: var(--warn); }
  #sub { font-size: .85rem; color: var(--dim); margin-top: .1rem; min-height: 1em;
         text-align: center; font-weight: bold; }

  /* ── stats bar (counts + version) ── */
  #stats-bar { width: 100%; max-width: 760px; margin-top: .9rem;
               display: flex; justify-content: space-between; align-items: baseline;
               font-size: .78rem; color: var(--dim); }
  #stats-bar b { color: var(--text); font-weight: bold; }
  #version-str { font-size: .68rem; color: #a0b0b8; font-style: italic; }

  /* ── event log table ── */
  #event-log { width: 100%; max-width: 760px; margin-top: 1.2rem;
               border-collapse: collapse; font-size: .78rem; }
  #event-log th { text-align: left; color: var(--dim); font-weight: normal;
                  letter-spacing: .06em; border-bottom: 1px solid var(--bord);
                  padding: .25rem .4rem; }
  #event-log td { padding: .22rem .4rem; border-bottom: 1px solid #e4e8ec;
                  font-family: var(--mono); }
  #event-log tr:first-child td { font-weight: bold; color: var(--text); }
  #event-log .cat-v { color: #0068a8; }
  #event-log .cat-p { color: #1a7a30; }
  #event-log tbody tr { background: var(--panel); cursor: pointer; }
  #event-log tbody tr:hover td { background: #eef4fb; }
  #event-log tbody tr.selected td { background: #ddeeff; font-weight: bold; }

  /* ── canvas panels ── */
  .chart-panel { width: 100%; max-width: 760px; margin-top: 1.4rem;
                 background: var(--panel); border: 1px solid var(--bord);
                 border-radius: 4px; padding: .5rem .5rem .3rem; }
  .chart-label { font-size: .7rem; color: var(--dim); letter-spacing: .1em;
                 text-transform: uppercase; margin-bottom: .25rem; padding-left: 52px; }
  .chart-label span { float: right; padding-right: 4px; text-transform: none;
                      letter-spacing: 0; }
  canvas { display: block; width: 100%; }

  /* ── metadata bar ── */
  #meta { margin-top: 1rem; font-size: .78rem; color: var(--dim);
          display: flex; gap: 1.4rem; flex-wrap: wrap; justify-content: center; }
  #meta b { color: var(--text); }
</style>
</head>
<body>

<div id="header">
  <div id="title"><span id="dot"></span>LIDAR_1</div>
  <div id="clock">--:--:--</div>
</div>

<div id="readout-bar">
  <span class="rdlabel">now:</span>
  <span id="readout-live" class="rdval">--.-- m</span>
  <span class="rdlabel" style="margin-left:1.4rem">last event:</span>
  <span id="readout-event" class="rdval">-- m</span>
</div>
<div id="sub">&nbsp;</div>

<div class="chart-panel">
  <div class="chart-label">Live — last 10 s<span id="lbl-live"></span></div>
  <canvas id="chart-live"></canvas>
</div>

<div class="chart-panel">
  <div class="chart-label" style="display:flex;align-items:baseline;justify-content:space-between;">
    <span><b>Last event</b><span id="lbl-event">— none yet —</span></span>
    <button id="btn-overlay" onclick="toggleOverlay()"
      style="font-size:.68rem;font-family:var(--mono);padding:.15rem .5rem;
             border:1px solid var(--bord);border-radius:3px;background:var(--panel);
             color:var(--dim);cursor:pointer;">overlay 5</button>
  </div>
  <canvas id="chart-event"></canvas>
</div>

<div id="meta">
  <span>baseline: <b id="m-bg">–</b> m</span>
  <span>σ: <b id="m-std">–</b> mm</span>
  <span>trigger: <b id="m-trig">–</b> m</span>
  <span>strength: <b id="m-str">–</b></span>
  <span>temp: <b id="m-temp">–</b> °C</span>
</div>

<table id="event-log">
  <thead><tr>
    <th>Time</th>
    <th>Dist (m)</th>
    <th>Dist RMS</th>
    <th>Category</th>
    <th>Dur (ms)</th>
    <th>Str avg</th>
    <th>Str RMS</th>
  </tr></thead>
  <tbody id="log-body"><tr><td colspan="7" style="color:var(--dim)">— no events yet —</td></tr></tbody>
</table>

<div id="stats-bar">
  <span>Last 60 min: <b id="cnt-60">–</b> &nbsp; Today: <b id="cnt-day">–</b></span>
  <span id="version-str">v<!--VER--></span>
</div>

<script>
// ── constants ────────────────────────────────────────────────────────────────
const WIN_S   = 10;
const MAX_PTS = 1000;
const MARGIN  = { l: 58, r: 8, t: 6, b: 28 };   // l wider for "MM.mmm m" labels

// ── state ────────────────────────────────────────────────────────────────────
const pts        = [];        // live history: {t, d, ev}
let   bgMean     = 0, bgTrig = 0;
let   evData        = null;   // currently displayed event (selected or latest)
let   evDataList    = [];     // last 10 full events (newest last)
let   lastEvId      = 0;
let   overlayOn     = false;
let   selectedEvIdx = null;   // index into evDataList of selected row (null = latest)

function toggleOverlay() {
  overlayOn = !overlayOn;
  document.getElementById('btn-overlay').style.fontWeight = overlayOn ? 'bold' : 'normal';
  document.getElementById('btn-overlay').style.color = overlayOn ? 'var(--accent)' : 'var(--dim)';
  drawEvent();
}

function selectEventRow(listIdx) {
  if (listIdx < 0 || listIdx >= evDataList.length) return;
  selectedEvIdx = listIdx;
  evData = evDataList[listIdx];
  updateChartLabel(evData);
  setLastEventReadout(evData.object_range_mm);
  highlightRow(listIdx);
  drawEvent();
}

function highlightRow(listIdx) {
  const tbody   = document.getElementById('log-body');
  const rows    = tbody.querySelectorAll('tr');
  const lastIdx = evDataList.length - 1;
  rows.forEach((tr, rowPos) => {
    tr.classList.toggle('selected', (lastIdx - rowPos) === listIdx);
  });
}

function updateChartLabel(ev) {
  const t   = new Date(ev.id * 1000);
  const hms = t.toTimeString().slice(0, 8);
  const ms  = t.getMilliseconds();
  const ts  = hms + '.' + Math.floor(ms / 100);
  document.getElementById('lbl-event').innerHTML =
    '&nbsp;&nbsp;<b>' + ts + '</b> · ' + (ev.category || '?') +
    ' · ' + ev.duration_ms + ' ms' +
    ' · <b style="color:#1a2530;font-weight:bold">' + (ev.object_range_mm / 1000).toFixed(3) + ' m</b>';
}

// ── canvas setup ─────────────────────────────────────────────────────────────
const cLive  = document.getElementById('chart-live');
const cEvent = document.getElementById('chart-event');
const xLive  = cLive.getContext('2d');
const xEvent = cEvent.getContext('2d');

function initCanvas(c, hpx) {
  const dpr = window.devicePixelRatio || 1;
  c.width   = c.offsetWidth * dpr;
  c.height  = hpx * dpr;
  c.style.height = hpx + 'px';
  c.getContext('2d').scale(dpr, dpr);
  return c.offsetWidth;
}

let CW = initCanvas(cLive,  220);
initCanvas(cEvent, 210);
window.addEventListener('resize', () => {
  CW = initCanvas(cLive,  220);
  initCanvas(cEvent, 210);
  drawEvent();
});

// ── shared drawing helpers ────────────────────────────────────────────────────
const FONT_SM  = '11px "Share Tech Mono", monospace';
const COL_GRID = '#d4dce4';
const COL_AXIS = '#8090a0';
const COL_TEXT = '#5a7080';

// Y axis labels in metres (MM.mmm), X labels in seconds.
// nSecTicks: how many whole-second grid lines to draw across [tMin,tMax].
function drawAxes(ctx, W, H, dMin, dMax, tMin, tMax, nSecTicks) {
  const ml = MARGIN.l, mr = MARGIN.r, mt = MARGIN.t, mb = MARGIN.b;
  const pw = W - ml - mr, ph = H - mt - mb;

  ctx.font        = FONT_SM;
  ctx.fillStyle   = COL_TEXT;
  ctx.strokeStyle = COL_GRID;
  ctx.lineWidth   = 1;

  // Y-axis grid + labels (metres, 3 dp)
  const dRange  = dMax - dMin || 1;
  const rawStep = dRange / 5;
  const mag     = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const yStep   = Math.ceil(rawStep / mag) * mag;
  const yFirst  = Math.ceil(dMin / yStep) * yStep;

  for (let d = yFirst; d <= dMax; d += yStep) {
    const y = mt + ph - (d - dMin) / dRange * ph;
    ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(W - mr, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText((d / 1000).toFixed(3), ml - 4, y);
  }

  // X-axis whole-second grid + labels
  for (let s = 0; s <= nSecTicks; s++) {
    const t = tMax - (nSecTicks - s);
    const x = ml + (t - tMin) / (tMax - tMin) * pw;
    ctx.beginPath(); ctx.moveTo(x, mt); ctx.lineTo(x, mt + ph); ctx.stroke();
    const lbl = (t - tMax).toFixed(0);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(lbl + 's', x, mt + ph + 4);
  }

  // Axis border
  ctx.strokeStyle = COL_AXIS;
  ctx.strokeRect(ml, mt, pw, ph);
}

const isSentinel = d => (d <= 0 || d >= 45000);  // lost-signal / saturation / error

// Plot a line skipping sentinel values; draws an × at the last valid y for each gap.
function plotLine(ctx, pts2d, W, H, dMin, dMax, tMin, tMax, color, glow) {
  const ml = MARGIN.l, mr = MARGIN.r, mt = MARGIN.t, mb = MARGIN.b;
  const pw = W - ml - mr, ph = H - mt - mb;
  const dRange = dMax - dMin || 1;
  const tRange = tMax - tMin || 1;
  const tx = t => ml + (t - tMin) / tRange * pw;
  const ty = d => mt + ph - (d - dMin) / dRange * ph;

  if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 4; }
  ctx.lineWidth   = 2;
  ctx.strokeStyle = color;

  let penDown = false;
  let lastValidY = null;
  const markers = [];

  ctx.beginPath();
  pts2d.forEach(([t, d]) => {
    const bad = isSentinel(d);
    const x = tx(t);
    if (bad) {
      if (penDown) ctx.stroke();
      penDown = false;
      ctx.beginPath();
      if (lastValidY !== null) markers.push({x, y: lastValidY});
    } else {
      const y = ty(d);
      if (!penDown) { ctx.moveTo(x, y); penDown = true; }
      else            ctx.lineTo(x, y);
      lastValidY = y;
    }
  });
  if (penDown) ctx.stroke();
  ctx.shadowBlur = 0;

  // × markers at sentinel positions
  const R = 4;
  ctx.strokeStyle = '#cc4400';
  ctx.lineWidth   = 1.5;
  markers.forEach(({x, y}) => {
    ctx.beginPath();
    ctx.moveTo(x-R, y-R); ctx.lineTo(x+R, y+R);
    ctx.moveTo(x+R, y-R); ctx.lineTo(x-R, y+R);
    ctx.stroke();
  });
  ctx.lineWidth = 2;
}

function drawDashedHLine(ctx, y, W, color) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.l, y); ctx.lineTo(W - MARGIN.r, y);
  ctx.stroke(); ctx.setLineDash([]); ctx.restore();
}

// ── live chart ───────────────────────────────────────────────────────────────
// pts entries: {t, lo, hi, ev}  — lo/hi are the min/max dist within the bucket;
// lo==hi==-1 means the bucket was all-sentinel values.
function drawLive() {
  const W = cLive.offsetWidth, H = cLive.offsetHeight;
  xLive.clearRect(0, 0, W, H);
  if (pts.length < 2) return;

  const now  = pts[pts.length - 1].t;
  const tMin = now - WIN_S;
  const vis  = pts.filter(p => p.t >= tMin);
  if (vis.length < 2) return;

  const validD = vis.filter(p => p.lo >= 0).flatMap(p => [p.lo, p.hi]);
  if (validD.length === 0) return;
  const dMin = Math.min(...validD, bgTrig  * 0.995) - 20;
  const dMax = Math.max(...validD, bgMean) * 1.002  + 20;
  const ml = MARGIN.l, mr = MARGIN.r, mt = MARGIN.t, mb = MARGIN.b;
  const pw = W - ml - mr, ph = H - mt - mb;
  const dRange = dMax - dMin || 1;
  const ty = d => mt + ph - (d - dMin) / dRange * ph;
  // Bucket half-width in px — rects touch adjacent buckets
  const BUCKET_S = 0.200;
  const bHalf = pw / WIN_S * BUCKET_S / 2;
  const tx = t => ml + (t - tMin) / WIN_S * pw;

  drawAxes(xLive, W, H, dMin, dMax, tMin, now, WIN_S);

  // Event highlight bands
  xLive.fillStyle = 'rgba(180,60,20,0.08)';
  let inEv = false, evX = 0;
  vis.forEach(p => {
    const x = tx(p.t);
    if (p.ev && !inEv) { inEv = true; evX = x; }
    if (!p.ev && inEv)  { inEv = false; xLive.fillRect(evX, mt, x - evX, ph); }
  });
  if (inEv) xLive.fillRect(evX, mt, tx(now) - evX, ph);

  // Reference lines
  drawDashedHLine(xLive, ty(bgMean), W, '#8090a0');
  drawDashedHLine(xLive, ty(bgTrig), W, '#c08080');

  // Solid filled rectangles spanning lo→hi, full bucket width
  const R = 4;
  vis.forEach(p => {
    const x = tx(p.t);
    if (p.lo < 0) {
      // All-sentinel bucket: × at chart midpoint
      const ym = mt + ph / 2;
      xLive.save();
      xLive.strokeStyle = '#cc4400'; xLive.lineWidth = 1.5;
      xLive.beginPath();
      xLive.moveTo(x-R, ym-R); xLive.lineTo(x+R, ym+R);
      xLive.moveTo(x+R, ym-R); xLive.lineTo(x-R, ym+R);
      xLive.stroke();
      xLive.restore();
      return;
    }
    const yLo = ty(p.lo), yHi = ty(p.hi);
    xLive.fillStyle = '#0068a8';
    xLive.fillRect(x - bHalf, yHi, bHalf * 2, Math.max(1, yLo - yHi));
  });

  // Y-axis label
  xLive.save();
  xLive.font = FONT_SM; xLive.fillStyle = COL_TEXT;
  xLive.textAlign = 'center'; xLive.textBaseline = 'middle';
  xLive.translate(10, mt + ph / 2);
  xLive.rotate(-Math.PI / 2);
  xLive.fillText('meters', 0, 0);
  xLive.restore();
}

// ── event chart ──────────────────────────────────────────────────────────────
// ── event chart ──────────────────────────────────────────────────────────────
// Overlay colour palette: index 0 = newest/most prominent
const OVERLAY_DIST_COLORS = ['#c05000','#b06828','#a07840','#908858','#809870'];
const OVERLAY_STR_COLORS  = ['rgba(26,122,48,0.55)','rgba(26,122,48,0.38)',
                              'rgba(26,122,48,0.28)','rgba(26,122,48,0.20)',
                              'rgba(26,122,48,0.14)'];

// Draw one event's strength + distance traces onto xEvent.
// toSec(offset_ms, ev) returns the X value (% of dur window).
function drawEventTrace(ev, toSec, tx, ty, sy, distColor, strColor) {
  const samps    = ev.samples;
  const validSet = new Set(ev.valid_indices || []);

  // Strength trace (drawn first so distance sits on top)
  xEvent.save();
  xEvent.strokeStyle = strColor;
  xEvent.lineWidth   = 1.0;
  xEvent.beginPath();
  let strPenDown = false;
  samps.forEach(s => {
    const t = toSec(s[0], ev), sv = s[2];
    const x = tx(t), y = sy(sv);
    if (y === null) {
      if (strPenDown) { xEvent.stroke(); xEvent.beginPath(); strPenDown = false; }
    } else {
      if (!strPenDown) { xEvent.moveTo(x, y); strPenDown = true; }
      else               xEvent.lineTo(x, y);
    }
  });
  if (strPenDown) xEvent.stroke();
  xEvent.restore();

  // Distance trace: thin (1px) outside valid region, thick (3px) inside
  xEvent.save();
  xEvent.strokeStyle = distColor;
  let penDown = false, lastWasValid = null;
  function flushPath() { if (penDown) { xEvent.stroke(); xEvent.beginPath(); penDown = false; } }
  samps.forEach((s, idx) => {
    const t = toSec(s[0], ev), d = s[1];
    const x = tx(t);
    const isValid = validSet.has(idx);
    if (isSentinel(d)) {
      flushPath(); lastWasValid = null;
    } else {
      const y = ty(d);
      if (lastWasValid !== null && isValid !== lastWasValid) flushPath();
      if (!penDown) {
        xEvent.lineWidth = isValid ? 3.0 : 1.0;
        xEvent.beginPath(); xEvent.moveTo(x, y); penDown = true;
      } else { xEvent.lineTo(x, y); }
      lastWasValid = isValid;
    }
  });
  flushPath();
  xEvent.restore();
}

function drawEvent() {
  const W = cEvent.offsetWidth, H = cEvent.offsetHeight;
  xEvent.clearRect(0, 0, W, H);
  if (!evData || !evData.samples.length) return;

  const baseIdx  = (selectedEvIdx !== null) ? selectedEvIdx : evDataList.length - 1;
  let   drawList;
  if (overlayOn && evDataList.length > 1) {
    const lo = Math.max(0, baseIdx - 4);
    drawList = evDataList.slice(lo, baseIdx + 1);
  } else {
    drawList = evData ? [evData] : [];
  }
  if (drawList.length === 0) return;

  // toSec: converts offset_ms to % of that event's dur window
  // 0% = dur_start_ms, 100% = dur_start_ms + duration_ms
  const toSec = (ms, ev) => (ms - (ev.dur_start_ms || 0)) / Math.max(ev.duration_ms || 1, 1) * 100;

  // Fixed X axis: -20% to +120%
  const tMin = -20, tMax = 120;

  // Compute Y axis ranges using only samples within the visible X window
  let allD = [], allS = [];
  drawList.forEach(ev => {
    ev.samples.forEach(s => {
      const pct = toSec(s[0], ev);
      if (pct < tMin || pct > tMax) return;   // outside visible window
      if (!isSentinel(s[1])) allD.push(s[1]);
      if (s[2] > 0)          allS.push(s[2]);
    });
  });
  if (allD.length === 0) return;

  const evBg      = evData.bg_mean;
  const dMax      = Math.max(...allD, evBg) + 30;
  const dMin      = Math.min(...allD) - Math.max(30, (dMax - Math.min(...allD)) * 0.05);
  const dRange    = dMax - dMin || 1;
  const sLogMin   = Math.log10(Math.max(1, Math.min(...allS) * 0.9));
  const sLogMax   = Math.log10(Math.max(...allS) * 1.1 + 1);
  const sLogRange = sLogMax - sLogMin || 1;

  const ml = MARGIN.l, mr = 52, mt = MARGIN.t, mb = MARGIN.b;
  const pw = W - ml - mr, ph = H - mt - mb;
  const tRange = tMax - tMin || 1;
  const tx = t  => ml + (t - tMin) / tRange * pw;
  const ty = d  => mt + ph - (d - dMin) / dRange * ph;
  const sy = sv => (sv > 0) ? mt + ph - (Math.log10(sv) - sLogMin) / sLogRange * ph : null;

  // ── grid ──
  xEvent.font        = FONT_SM;
  xEvent.fillStyle   = COL_TEXT;
  xEvent.strokeStyle = COL_GRID;
  xEvent.lineWidth   = 1;

  // Left Y-axis: distance in integer metres
  const rawStep = (dMax - dMin) / 5;
  const mag     = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const yStep   = Math.ceil(rawStep / mag) * mag;
  const yFirst  = Math.ceil(dMin / yStep) * yStep;
  for (let d = yFirst; d <= dMax; d += yStep) {
    const y = ty(d);
    xEvent.beginPath(); xEvent.moveTo(ml, y); xEvent.lineTo(W - mr, y); xEvent.stroke();
    xEvent.textAlign = 'right'; xEvent.textBaseline = 'middle';
    xEvent.fillText(Math.round(d / 1000), ml - 4, y);
  }

  // X-axis: ticks every 20%
  const TICK_PCT  = 20;
  const firstTick = Math.ceil(tMin / TICK_PCT) * TICK_PCT;
  for (let t = firstTick; t <= tMax + 1e-9; t += TICK_PCT) {
    const x = tx(t);
    xEvent.beginPath(); xEvent.moveTo(x, mt); xEvent.lineTo(x, mt + ph); xEvent.stroke();
    xEvent.textAlign = 'center'; xEvent.textBaseline = 'top';
    xEvent.fillText(Math.round(t) + '%', x, mt + ph + 4);
  }

  // Right Y-axis: strength log labels
  xEvent.fillStyle = '#1a7a30';
  const sDecMin = Math.floor(sLogMin), sDecMax = Math.ceil(sLogMax);
  const logTicks = [];
  for (let dec = sDecMin; dec <= sDecMax; dec++) {
    [1, 2, 5].forEach(m => {
      const v = m * Math.pow(10, dec);
      if (v >= Math.pow(10, sLogMin) * 0.99 && v <= Math.pow(10, sLogMax) * 1.01) logTicks.push(v);
    });
  }
  logTicks.forEach(sv => {
    const y = sy(sv);
    if (y === null || y < mt || y > mt + ph) return;
    xEvent.textAlign = 'left'; xEvent.textBaseline = 'middle';
    xEvent.fillText(sv >= 1000 ? (sv/1000).toFixed(0)+'k' : Math.round(sv), W - mr + 4, y);
    xEvent.strokeStyle = '#1a7a30'; xEvent.lineWidth = 1;
    xEvent.beginPath(); xEvent.moveTo(W - mr, y); xEvent.lineTo(W - mr + 4, y); xEvent.stroke();
  });

  xEvent.strokeStyle = COL_AXIS;
  xEvent.strokeRect(ml, mt, pw, ph);

  // Baseline + object range reference lines (most recent event)
  drawDashedHLine(xEvent, ty(evBg), W - mr + MARGIN.r, '#8090a0');
  if (evData.object_range_mm)
    drawDashedHLine(xEvent, ty(evData.object_range_mm), W - mr + MARGIN.r, '#a0a0a0');

  // 0% and 100% event boundary lines
  xEvent.save();
  xEvent.strokeStyle = '#b0b8c0'; xEvent.lineWidth = 1; xEvent.setLineDash([3, 3]);
  [0, 100].forEach(pct => {
    const x = tx(pct);
    xEvent.beginPath(); xEvent.moveTo(x, mt); xEvent.lineTo(x, mt + ph); xEvent.stroke();
  });
  xEvent.setLineDash([]); xEvent.restore();

  // Draw traces oldest-first so newest is on top, clipped to plot area
  xEvent.save();
  xEvent.beginPath();
  xEvent.rect(ml, mt, pw, ph);
  xEvent.clip();
  const n = drawList.length;
  for (let i = 0; i < n; i++) {
    const ci = (n - 1) - i;
    drawEventTrace(drawList[i], toSec, tx, ty, sy,
                   OVERLAY_DIST_COLORS[Math.min(ci, OVERLAY_DIST_COLORS.length - 1)],
                   OVERLAY_STR_COLORS [Math.min(ci, OVERLAY_STR_COLORS.length  - 1)]);
  }
  xEvent.restore();

  // Axis labels
  xEvent.save();
  xEvent.font = FONT_SM; xEvent.fillStyle = COL_TEXT;
  xEvent.textAlign = 'center'; xEvent.textBaseline = 'middle';
  xEvent.translate(10, mt + ph / 2); xEvent.rotate(-Math.PI / 2);
  xEvent.fillText('meters', 0, 0);
  xEvent.restore();

  xEvent.save();
  xEvent.font = FONT_SM; xEvent.fillStyle = '#1a7a30';
  xEvent.textAlign = 'center'; xEvent.textBaseline = 'middle';
  xEvent.translate(W - 10, mt + ph / 2); xEvent.rotate(Math.PI / 2);
  xEvent.fillText('signal level', 0, 0);
  xEvent.restore();
}

// Render live chart at 30 fps; event chart only redrawn when new data arrives
setInterval(drawLive, 33);

// ── clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const hh  = String(now.getHours()).padStart(2,'0');
  const mm  = String(now.getMinutes()).padStart(2,'0');
  const ss  = String(now.getSeconds()).padStart(2,'0');
  document.getElementById('clock').textContent = hh + ':' + mm + ':' + ss;
}
updateClock();
setInterval(updateClock, 1000);

// ── distance readout ─────────────────────────────────────────────────────────
function setReadout(dist_mm, isEv) {
  const el = document.getElementById('readout-live');
  el.textContent = (dist_mm / 1000).toFixed(2) + ' m';
  el.className = 'rdval' + (isEv ? ' ev' : '');
}
function setLastEventReadout(range_mm) {
  document.getElementById('readout-event').textContent = (range_mm / 1000).toFixed(3) + ' m';
}

// ── SSE ──────────────────────────────────────────────────────────────────────
const src  = new EventSource('/stream');
const dot  = document.getElementById('dot');
const sub  = document.getElementById('sub');

src.onmessage = e => {
  const msg = JSON.parse(e.data);
  bgMean = msg.bg_mean;
  bgTrig = msg.trigger;

  // Append bucketed history points {t, lo, hi, ev}
  msg.history.forEach(([t, lo, hi, ev]) => {
    if (pts.length && pts[pts.length - 1].t >= t) return;
    pts.push({t, lo, hi, ev: !!ev});
  });
  while (pts.length > MAX_PTS) pts.shift();

  setReadout(msg.dist_avg ?? msg.dist, msg.in_event);
  sub.textContent = msg.in_event ? (msg.category || 'EVENT IN PROGRESS') : '';

  document.getElementById('m-bg').textContent   = (msg.bg_mean / 1000).toFixed(3);
  document.getElementById('m-std').textContent  = msg.bg_std.toFixed(1);
  document.getElementById('m-trig').textContent = (msg.trigger / 1000).toFixed(3);
  document.getElementById('m-str').textContent  = msg.strength;
  document.getElementById('m-temp').textContent = msg.temp_c.toFixed(1);

  // Check for new completed event
  if (msg.last_event_id && msg.last_event_id !== lastEvId) {
    lastEvId = msg.last_event_id;
    fetch('/events')
      .then(r => r.json())
      .then(list => {
        evDataList = list;
        if (selectedEvIdx === null || selectedEvIdx === evDataList.length - 2) {
          selectedEvIdx = null;
          evData = list.length ? list[list.length - 1] : null;
        }
        if (evData) {
          updateChartLabel(evData);
          setLastEventReadout(evData.object_range_mm);
        }
        rebuildTable(msg.event_log);
        drawEvent();
      });
  }

  // Event counts
  if (msg.count_60min !== undefined)
    document.getElementById('cnt-60').textContent  = msg.count_60min;
  if (msg.count_today !== undefined)
    document.getElementById('cnt-day').textContent = msg.count_today;

  // Table only rebuilt when a new event arrives (handled above); no-op here.

  dot.className = 'live';   // CSS @keyframes handles 1 Hz blink; set once, stays
};

src.onerror = () => { dot.className = ''; };

// ── table rendering ───────────────────────────────────────────────────────────
function rebuildTable(eventLog) {
  if (!eventLog || !eventLog.length) return;
  const rows    = eventLog.slice().reverse();
  const tbody   = document.getElementById('log-body');
  const lastIdx = evDataList.length - 1;
  tbody.innerHTML = rows.map((ev, rowPos) => {
    const evIdx   = lastIdx - rowPos;
    const dt      = new Date(ev.epoch * 1000);
    const hms     = dt.toTimeString().slice(0, 8);
    const s1      = Math.floor(dt.getMilliseconds() / 100);
    const ts      = hms + '.' + s1;
    const distM   = (ev.range_mm / 1000).toFixed(2);
    const cat     = ev.category || '?';
    const catCls  = cat === 'vehicle' ? 'cat-v' : 'cat-p';
    const distRms = ev.range_rms !== undefined ? (ev.range_rms / 1000).toFixed(3) : '–';
    const selIdx  = (selectedEvIdx !== null) ? selectedEvIdx : lastIdx;
    const selCls  = (evIdx === selIdx && evIdx >= 0) ? ' selected' : '';
    const onclick = evIdx >= 0 ? `onclick="selectEventRow(${evIdx})"` : '';
    return `<tr class="${selCls}" ${onclick}>
      <td>${ts}</td>
      <td>${distM}</td>
      <td>${distRms}</td>
      <td class="${catCls}">${cat}</td>
      <td>${ev.duration_ms}</td>
      <td>${ev.str_avg}</td>
      <td>${ev.str_rms}</td>
    </tr>`;
  }).join('');
}

// ── keyboard navigation ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (!evDataList.length) return;
  const curIdx = (selectedEvIdx !== null) ? selectedEvIdx : evDataList.length - 1;
  if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
    e.preventDefault();
    selectEventRow(Math.min(curIdx + 1, evDataList.length - 1));
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
    e.preventDefault();
    selectEventRow(Math.max(curIdx - 1, 0));
  }
});
</script>
</body>
</html>"""

@_flask_app.route('/')
def _index():
    page = _PAGE_HTML.replace('<!--VER-->', VERSION)
    return Response(page, mimetype='text/html')

@_flask_app.route('/event')
def _event():
    with _web_lock:
        ev = _last_event
    if ev is None:
        return Response('{}', mimetype='application/json')
    return Response(json.dumps(ev), mimetype='application/json')

@_flask_app.route('/events')
def _events():
    with _web_lock:
        evs = list(_recent_events)
    return Response(json.dumps(evs), mimetype='application/json')

@_flask_app.route('/stream')
def _stream():
    def gen():
        interval = 1.0 / WEB_HZ
        while True:
            time.sleep(interval)
            with _web_lock:
                snap = dict(_web_state)
                # Build 200 ms min/max buckets over the full history window.
                # Each bucket: [t_center, d_min, d_max, any_ev]
                # Sentinel values (0, 45000 mm) are excluded from min/max;
                # an all-sentinel bucket is flagged with d_min=d_max=-1.
                BUCKET_S  = 0.200
                SENTINELS = {0, 45000}
                history   = []
                last_bucket_mean = None
                if _web_buf:
                    buf_list  = list(_web_buf)
                    t_end     = buf_list[-1][0]
                    t_start   = t_end - WEB_BUF_SECS
                    t_bucket  = t_start
                    i = 0
                    n = len(buf_list)
                    while t_bucket < t_end:
                        t_next = t_bucket + BUCKET_S
                        vals, evs = [], []
                        while i < n and buf_list[i][0] < t_next:
                            bt, bd, bs, bev = buf_list[i]
                            if bt >= t_bucket:
                                if 0 < bd < 45000:   # exclude 0, >=45000 (lost/saturated/error)
                                    vals.append(bd)
                                evs.append(bev)
                            i += 1
                        if vals:
                            bmean = sum(vals) / len(vals)
                            last_bucket_mean = bmean
                            history.append([round(t_bucket + BUCKET_S/2, 3),
                                            min(vals), max(vals), int(any(evs))])
                        elif evs:   # all-sentinel bucket
                            history.append([round(t_bucket + BUCKET_S/2, 3),
                                            -1, -1, int(any(evs))])
                        t_bucket = t_next
                snap['dist_avg'] = round(last_bucket_mean) if last_bucket_mean is not None \
                                   else snap.get('dist', 0)
                snap['event_log'] = list(_event_log)
                # Counts computed fresh each tick so midnight rollover is handled
                now_t = time.time()
                midnight = _dt.datetime.combine(
                    _dt.date.today(), _dt.time.min).timestamp()
                snap['count_60min'] = sum(1 for e in _event_epochs
                                          if now_t - e <= 3600)
                snap['count_today'] = sum(1 for e in _event_epochs
                                          if e >= midnight)
            snap['history'] = history
            payload = json.dumps(snap)
            yield f"data: {payload}\n\n"
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})

def _start_web_server():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)   # suppress per-request noise
    _flask_app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)

threading.Thread(target=_start_web_server, daemon=True, name='flask').start()
print(f"Web interface starting on http://0.0.0.0:{WEB_PORT}/  (LAN accessible)")

# --- Setup ---
print("TF02-Pro LIDAR Event Logger  Version", VERSION)

PORT = find_ch341_port()  # auto select the right port (assuming no other similar devices)
# PORT          = '/dev/ttyUSB1'

os.makedirs(OUTPUT_DIR, exist_ok=True)
ser = serial.Serial(PORT, 115200, timeout=0.1)

# Switch TF02-Pro to mm output mode (command: 5A 05 05 06 6A)
ser.write(bytes([0x5A, 0x05, 0x05, 0x06, 0x6A]))
time.sleep(0.1)
ser.reset_input_buffer()  # discard any frames that arrived before mode switch

def read_sample():
    """Block until one valid 9-byte TF02-Pro frame is parsed. Returns (dist, strength, temp_c)."""
    while True:
        time.sleep(0.005) # avoid busy loop; TF02-Pro sends at ~100 Hz, so this is a short wait
        if ser.in_waiting >= 9:
            if ser.read() == b'Y':
                if ser.read() == b'Y':
                    dist_l = ord(ser.read())
                    dist_h = ord(ser.read())
                    str_l  = ord(ser.read())
                    str_h  = ord(ser.read())
                    temp_l = ord(ser.read())
                    temp_h = ord(ser.read())
                    _      = ser.read()   # checksum (ignored)
                    dist     = dist_h * 256 + dist_l
                    strength = str_h  * 256 + str_l
                    temp_c   = (temp_h * 256 + temp_l) / 8.0 - 256.0
                    return dist, strength, temp_c

# ---------------------------------------------------------------------------
# Phase 1: seed baseline from BASELINE_SECS seconds of quiet readings
# Restarts if any sample deviates more than BASELINE_MAX_RANGE mm from any other.
# ---------------------------------------------------------------------------
BASELINE_MAX_RANGE = 600   # mm: max(dist) - min(dist) threshold to restart collection
n_seed = BASELINE_SECS * SAMPLE_RATE
seed_dist = []
last_baseline_print = 0.0
seed_min = seed_max = None

def _start_seed():
    global seed_dist, seed_min, seed_max, last_baseline_print
    seed_dist = []
    seed_min = seed_max = None
    last_baseline_print = 0.0
    print(f"Collecting {n_seed} samples ({BASELINE_SECS}s) to seed baseline...")

_start_seed()
while len(seed_dist) < n_seed:
    d, s, temp_c = read_sample()
    if seed_min is None:
        seed_min = seed_max = d
    else:
        seed_min = min(seed_min, d)
        seed_max = max(seed_max, d)
    if seed_max - seed_min > BASELINE_MAX_RANGE:
        print(f"  baseline disturbed (range={seed_max-seed_min} mm) — restarting")
        _start_seed()
        continue
    seed_dist.append(d)
    t = time.time()
    if t - last_baseline_print >= 0.5:
        print(f"  baseline sample: dist={d} mm  strength={s}  temp={temp_c:.1f} C")
        last_baseline_print = t

bg_mean = float(np.mean(seed_dist))
bg_std  = float(np.std(seed_dist))
trigger_thresh = bg_mean - TRIGGER_SIGMA * bg_std - TRIGGER_OFFSET

print(f"Baseline: mean={bg_mean:.2f} mm  std={bg_std:.3f} mm")
print(f"Trigger threshold: < {trigger_thresh:.2f} mm")
print(f"Writing events to: {os.path.abspath(OUTPUT_DIR)}/")

# ---------------------------------------------------------------------------
# Phase 2: live detection loop
# ---------------------------------------------------------------------------
pre_buf = collections.deque(maxlen=PRE_TRIGGER)  # rolling (timestamp, dist, strength)
below_count  = 0          # consecutive samples below trigger threshold
in_event     = False
rearm_count  = 0          # consecutive samples back within baseline
event_samples = []        # list of (offset_ms, dist, strength) for current event
event_t0      = None      # absolute time of first event sample (after pre-trigger)
event_start_epoch = None

MAX_EVENT_SAMPLES = int(MAX_EVENT_SECS * SAMPLE_RATE)
REARM_SAMPLES     = int(REARM_SECS    * SAMPLE_RATE)
last_status_time  = time.time()
last_event_print_time = 0.0

def analyze_event(samples, bg_mean):
    """
    Classify event and extract metrics using rise-time ratio.
    Returns dict with keys: category, object_range_mm, duration_ms, rise_ratio
    """
    times     = np.array([s[0] for s in samples], dtype=float)
    dists     = np.array([s[1] for s in samples], dtype=float)
    strengths = np.array([s[2] for s in samples], dtype=float)

    # --- Clean dropouts: replace sentinels with linearly interpolated values ---
    dropout_mask = np.isin(dists, list(DROPOUT_VALUES))
    if dropout_mask.any():
        dists[dropout_mask] = np.nan
        nans = np.isnan(dists)
        idx  = np.arange(len(dists))
        dists[nans] = np.interp(idx[nans], idx[~nans], dists[~nans])

    excursion = bg_mean - dists          # positive when object present
    if excursion.max() <= 0:
        return None                      # no real event

    # --- Step 1: stable_mask — three consecutive backward diffs all below threshold ---
    # Point i is stable if |dists[i]-dists[i-1]| < threshold for i, i-1, and i-2.
    # First two points can never qualify (insufficient history).
    bdiff = np.abs(np.diff(dists))          # len = N-1; bdiff[i] = |dists[i+1]-dists[i]|
    stable_mask = np.zeros(len(dists), dtype=bool)
    for i in range(3, len(dists)):
        if (bdiff[i-1] < STABLE_RATE_THRESHOLD and
            bdiff[i-2] < STABLE_RATE_THRESHOLD and
            bdiff[i-3] < STABLE_RATE_THRESHOLD):
            stable_mask[i] = True

    # --- Step 2: max_exc from stable points only; fall back to all points if none ---
    if stable_mask.any():
        max_exc = float((bg_mean - dists[stable_mask]).max())
    else:
        max_exc = float(excursion.max())
    if max_exc <= 0:
        return None

    # --- Rise-time ratio classifier (uses full excursion array, not just stable) ---
    above_lo = np.where(excursion >= 0.10 * max_exc)[0]
    above_hi = np.where(excursion >= 0.90 * max_exc)[0]
    if len(above_lo) == 0 or len(above_hi) == 0:
        rise_ratio = 1.0                 # degenerate — treat as pedestrian
    else:
        lead_time  = times[above_hi[0]]  - times[above_lo[0]]   # 10%→90% on entry
        trail_time = times[above_lo[-1]] - times[above_hi[-1]]  # 90%→10% on exit
        total_time = times[above_lo[-1]] - times[above_lo[0]]   # full width at 10%
        rise_ratio = float((lead_time + trail_time) / total_time) if total_time > 0 else 1.0

    category = "pedestrian" if rise_ratio >= RISE_TIME_THRESHOLD else "vehicle"

    # --- Object range ---
    # Strength-weighted mean of points that are both stable and have >10% excursion.
    mask = stable_mask & (excursion > 0.10 * max_exc)
    if mask.any():
        object_range = float(np.average(dists[mask], weights=strengths[mask]))
    else:
        object_range = float(dists[np.argmin(dists)])  # degenerate fallback

    # --- Duration: time spent with excursion > 50% of nominal excursion ---
    nominal_exc = bg_mean - object_range
    dur_mask    = excursion > 0.50 * nominal_exc
    if dur_mask.any():
        dur_ms       = int(times[dur_mask][-1] - times[dur_mask][0])
        dur_start_ms = int(times[dur_mask][0])
    else:
        dur_ms = 0
        dur_start_ms = int(times[0])

    # --- RMS deviation of valid points from weighted-mean distance ---
    if mask.any():
        deviations  = dists[mask] - object_range
        range_rms   = float(np.sqrt(np.mean(deviations ** 2)))
    else:
        range_rms = 0.0

    return {
        "category":        category,
        "object_range_mm": round(object_range),
        "range_rms_mm":    round(range_rms, 1),
        "duration_ms":     dur_ms,
        "dur_start_ms":    dur_start_ms,
        "rise_ratio":      round(rise_ratio, 3),
        "valid_indices":   [int(i) for i in np.where(mask)[0]],
    }
def save_event(start_epoch, samples, result):
    _TZ = pytz.timezone('America/Los_Angeles')
    dt  = datetime.fromtimestamp(start_epoch, tz=_TZ)
    ms  = int(round((start_epoch % 1) * 1000))
    ts  = dt.strftime('%Y%m%d_%H%M%S') + f'_{ms:03d}'
    fname = os.path.join(OUTPUT_DIR, f"event_{ts}.csv")
    with open(fname, 'w') as f:
        hms_string = datetime.fromtimestamp(start_epoch).strftime('%H:%M:%S.%f')[:-3]
        f.write(f"# start={hms_string} epoch={start_epoch:.3f} bg_mean={bg_mean:.2f} bg_std={bg_std:.3f} units=mm\n")
        if result:
            f.write(f"# category={result['category']}  object_range={result['object_range_mm']}mm"
                    f"  range_rms={result['range_rms_mm']}mm"
                    f"  duration={result['duration_ms']}ms  rise_ratio={result['rise_ratio']}\n")
        f.write("offset_ms,dist_mm,strength\n")
        for offset_ms, dist, strength in samples:
            f.write(f"{offset_ms},{dist},{strength}\n")
    print(f"  saved {fname}  ({len(samples)} samples)")
    if result:
        print(f"  category={result['category']}  object_range={result['object_range_mm']}mm"
              f"  range_rms={result['range_rms_mm']}mm"
              f"  duration={result['duration_ms']}ms  rise_ratio={result['rise_ratio']}")

sample_idx = 0   # global sample counter used for ms offsets within events

while True:
    d, strength, temp_c = read_sample()
    t_now = time.time()
    sample_idx += 1

    trigger_thresh = bg_mean - TRIGGER_SIGMA * bg_std - TRIGGER_OFFSET
    _web_push(t_now, d, strength, temp_c, in_event)

    if not in_event:
        pre_buf.append((t_now, d, strength))

        # Check trigger: N consecutive samples below threshold
        if d < trigger_thresh:
            below_count += 1
        else:
            below_count = 0

        if below_count >= TRIGGER_N:
            in_event = True
            rearm_count = 0
            event_samples = []

            # Prepend pre-trigger buffer (all of it; includes the triggering samples)
            t_pre0 = pre_buf[0][0]
            for tb, db, sb in pre_buf:
                offset_ms = round((tb - t_pre0) * 1000)
                event_samples.append((offset_ms, db, sb))

            # ms offset of the first sample that crossed below threshold
            trigger_ms = round((list(pre_buf)[-TRIGGER_N][0] - t_pre0) * 1000)

            event_start_epoch = t_pre0
            t_str = f"{time.strftime('%H:%M:%S', time.localtime(t_now))}.{int(t_now % 1 * 1000):03d}"
            print(f"EVENT triggered at {t_str}  dist={d}  threshold={trigger_thresh:.1f}")

        else:
            # Update slow EMA baseline only when signal is quiet
            if bg_std > 0 and abs(d - bg_mean) < QUIET_STD_MULT * bg_std:
                bg_mean = EMA_ALPHA * d + (1 - EMA_ALPHA) * bg_mean
                # Update bg_std as EMA of squared deviation
                bg_std = float(np.sqrt(
                    EMA_ALPHA * (d - bg_mean) ** 2 + (1 - EMA_ALPHA) * bg_std ** 2
                ))

        if t_now - last_status_time >= 60.0:
            print(f"[{time.strftime('%H:%M:%S')}] baseline={bg_mean:.2f} mm  std={bg_std:.3f} mm  trigger<{trigger_thresh:.2f} mm  temp={temp_c:.1f} C")
            last_status_time = t_now

    else:
        # Recording event
        offset_ms = round((t_now - event_start_epoch) * 1000)
        event_samples.append((offset_ms, d, strength))

        if t_now - last_event_print_time >= 0.5:
            print(f"  +{offset_ms}ms  dist={d} mm  strength={strength}")
            last_event_print_time = t_now

        # Check re-arm condition: sustained return to baseline
        if abs(d - bg_mean) <= REARM_DIFFERENCE:
            rearm_count += 1
        else:
            rearm_count = 0

        end_reason = None
        if rearm_count >= REARM_SAMPLES:
            end_reason = "rearm"
        elif len(event_samples) >= MAX_EVENT_SAMPLES:
            end_reason = "timeout"

        if end_reason:
            print(f"  event ended ({end_reason})  {len(event_samples)} samples")
            result = analyze_event(event_samples, bg_mean)
            save_event(event_start_epoch, event_samples, result)
            # Compute strength stats over valid (non-sentinel) dist samples
            strengths = [s[2] for s in event_samples if 0 < s[1] < 45000]
            str_avg = round(sum(strengths) / len(strengths), 1) if strengths else 0
            str_rms = round((sum(x*x for x in strengths) / len(strengths)) ** 0.5, 1) if strengths else 0
            log_entry = {
                "epoch":      event_start_epoch,
                "category":   result['category']        if result else '',
                "range_mm":   result['object_range_mm'] if result else 0,
                "range_rms":  result['range_rms_mm']    if result else 0,
                "duration_ms":result['duration_ms']     if result else 0,
                "str_avg":    str_avg,
                "str_rms":    str_rms,
            }
            with _web_lock:
                _web_state['category'] = result['category'] if result else ''
                _web_state['last_event_id'] = event_start_epoch
                _last_event = {
                    "id":             event_start_epoch,
                    "category":       result['category']        if result else '',
                    "object_range_mm":result['object_range_mm'] if result else 0,
                    "duration_ms":    result['duration_ms']     if result else 0,
                    "dur_start_ms":   result['dur_start_ms']    if result else 0,
                    "bg_mean":        bg_mean,
                    "trigger_ms":     trigger_ms,
                    "valid_indices":  result['valid_indices']   if result else [],
                    # Full 100-sps samples: [offset_ms, dist_mm, strength]
                    "samples":        [[s[0], s[1], s[2]] for s in event_samples],
                }
                _recent_events.append(_last_event)
                _event_log.append(log_entry)
                _event_epochs.append(event_start_epoch)
            in_event = False
            below_count = 0
            pre_buf.clear()
