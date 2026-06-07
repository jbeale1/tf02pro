#!/usr/bin/env python3

# TF02-Pro LIDAR event logger
# Detects objects crossing the beam and saves per-event CSVs.
# J.Beale  5-Jun-2026

import serial
import numpy as np
import time
from datetime import datetime
import collections
import os
import threading
import json
import serial.tools.list_ports   # enable port scan for correct device
from flask import Flask, Response, render_template_string


# --- Configuration ---
VERSION = "1.015  2026-06-07"  # bold event label/time, SS.S timestamp, 4s baseline
OUTPUT_DIR    = 'events'

WEB_PORT      = 8080         # LAN web interface port
WEB_HZ        = 5           # SSE update rate for browser clients
WEB_BUF_SECS  = 50          # seconds of history kept for web display (10s shown)

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
_last_event = None   # dict: {id, category, bg_mean, samples: [[offset_ms, dist], ...]}

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
         min-height: 100vh; padding: 1.2rem 1rem 2rem; }

  /* ── header row ── */
  #header { width: 100%; max-width: 760px; display: flex;
            align-items: baseline; justify-content: space-between;
            margin-bottom: 1.2rem; }
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

  /* ── distance readout ── */
  #readout { font-family: var(--mono); font-size: clamp(3rem,14vw,5.5rem);
             color: var(--accent); transition: color .15s;
             display: flex; align-items: baseline; gap: 0; }
  #readout.ev { color: var(--warn); }
  .rdig { display: inline-block; width: .62em; text-align: center; }
  .rsep { display: inline-block; width: .28em; text-align: center; }
  #r-unit { font-size: .38em; color: var(--dim); margin-left: .5rem;
            font-family: var(--mono); align-self: flex-end; padding-bottom: .25em; }
  #sub { font-size: .85rem; color: var(--dim); margin-top: .25rem; min-height: 1.2em;
         text-align: center; font-weight: bold; }

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

<!-- mm.mm: 2 integer digits + dot + 2 fractional digits -->
<div id="readout">
  <span class="rdig" id="r0">-</span>
  <span class="rdig" id="r1">-</span>
  <span class="rsep">.</span>
  <span class="rdig" id="r2">-</span>
  <span class="rdig" id="r3">-</span>
  <span id="r-unit">m</span>
</div>
<div id="sub">&nbsp;</div>

<div class="chart-panel">
  <div class="chart-label">Live — last 10 s<span id="lbl-live"></span></div>
  <canvas id="chart-live"></canvas>
</div>

<div class="chart-panel">
  <div class="chart-label"><b>Last event</b><span id="lbl-event">— none yet —</span></div>
  <canvas id="chart-event"></canvas>
</div>

<div id="meta">
  <span>baseline: <b id="m-bg">–</b> m</span>
  <span>σ: <b id="m-std">–</b> mm</span>
  <span>trigger: <b id="m-trig">–</b> m</span>
  <span>strength: <b id="m-str">–</b></span>
  <span>temp: <b id="m-temp">–</b> °C</span>
</div>

<script>
// ── constants ────────────────────────────────────────────────────────────────
const WIN_S   = 10;
const MAX_PTS = 1000;
const MARGIN  = { l: 58, r: 8, t: 6, b: 28 };   // l wider for "MM.mmm m" labels

// ── state ────────────────────────────────────────────────────────────────────
const pts      = [];          // live history: {t, d, ev}
let   bgMean   = 0, bgTrig = 0;
let   evData   = null;        // last event: {id, category, bg_mean, samples}
let   lastEvId = 0;

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
  xLive.fillText('m', 0, 0);
  xLive.restore();
}

// ── event chart ──────────────────────────────────────────────────────────────
function drawEvent() {
  const W = cEvent.offsetWidth, H = cEvent.offsetHeight;
  xEvent.clearRect(0, 0, W, H);
  if (!evData || !evData.samples.length) return;

  const samps  = evData.samples;            // [[offset_ms, dist, strength], ...]
  const trigMs = evData.trigger_ms || 0;
  const allD   = samps.map(s => s[1]).filter(d => !isSentinel(d));
  if (allD.length === 0) return;
  const evBg   = evData.bg_mean;
  const dMin   = Math.min(...allD) - 30;
  const dMax   = Math.max(...allD, evBg) + 30;
  const dRange = dMax - dMin || 1;

  // Strength axis — use all samples (sentinels may have odd strength, include anyway)
  const allS  = samps.map(s => s[2] || 0);
  const sMin  = Math.max(0, Math.min(...allS) * 0.95);
  const sMax  = Math.max(...allS) * 1.05 + 1;
  const sRange = sMax - sMin || 1;

  // t=0 at trigger crossing
  const toSec = ms => (ms - trigMs) / 1000;
  const tMin  = toSec(samps[0][0]);
  const tMax  = toSec(samps[samps.length - 1][0]);

  // Use wider right margin to fit strength axis labels
  const ml = MARGIN.l, mr = 52, mt = MARGIN.t, mb = MARGIN.b;
  const pw = W - ml - mr, ph = H - mt - mb;
  const tRange = tMax - tMin || 1;
  const tx = t  => ml + (t - tMin) / tRange * pw;
  const ty = d  => mt + ph - (d - dMin) / dRange * ph;
  const sy = sv => mt + ph - (sv - sMin) / sRange * ph;

  // ── grid ──
  xEvent.font        = FONT_SM;
  xEvent.fillStyle   = COL_TEXT;
  xEvent.strokeStyle = COL_GRID;
  xEvent.lineWidth   = 1;

  // Left Y-axis: distance in metres
  const rawStep = (dMax - dMin) / 5;
  const mag     = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const yStep   = Math.ceil(rawStep / mag) * mag;
  const yFirst  = Math.ceil(dMin / yStep) * yStep;
  for (let d = yFirst; d <= dMax; d += yStep) {
    const y = ty(d);
    xEvent.beginPath(); xEvent.moveTo(ml, y); xEvent.lineTo(W - mr, y); xEvent.stroke();
    xEvent.textAlign = 'right'; xEvent.textBaseline = 'middle';
    xEvent.fillText((d / 1000).toFixed(3), ml - 4, y);
  }

  // X-axis: 200 ms ticks
  const TICK_S    = 0.200;
  const firstTick = Math.ceil(tMin / TICK_S) * TICK_S;
  for (let t = firstTick; t <= tMax + 1e-9; t += TICK_S) {
    const x = tx(t);
    xEvent.beginPath(); xEvent.moveTo(x, mt); xEvent.lineTo(x, mt + ph); xEvent.stroke();
    xEvent.textAlign = 'center'; xEvent.textBaseline = 'top';
    xEvent.fillText(t.toFixed(1) + 's', x, mt + ph + 4);
  }

  // Right Y-axis: strength labels (no grid lines — left axis already drew them)
  const sRawStep = sRange / 5;
  const sMag     = Math.pow(10, Math.floor(Math.log10(Math.max(sRawStep, 1))));
  const ssStep   = Math.ceil(sRawStep / sMag) * sMag || 1;
  const ssFirst  = Math.ceil(sMin / ssStep) * ssStep;
  xEvent.fillStyle = '#1a7a30';
  for (let sv = ssFirst; sv <= sMax; sv += ssStep) {
    const y = sy(sv);
    xEvent.textAlign = 'left'; xEvent.textBaseline = 'middle';
    xEvent.fillText(Math.round(sv), W - mr + 4, y);
  }

  // Axis border
  xEvent.strokeStyle = COL_AXIS;
  xEvent.strokeRect(ml, mt, pw, ph);

  // Baseline dashed line
  drawDashedHLine(xEvent, ty(evBg), W - mr + MARGIN.r, '#8090a0');

  // t=0 trigger line
  xEvent.save();
  xEvent.strokeStyle = '#4080c0'; xEvent.lineWidth = 1.5;
  xEvent.beginPath(); xEvent.moveTo(tx(0), mt); xEvent.lineTo(tx(0), mt + ph);
  xEvent.stroke(); xEvent.restore();

  // ── strength trace (dotted, secondary, drawn first so distance is on top) ──
  const strPts = samps.map(s => [toSec(s[0]), s[2] || 0]);
  xEvent.save();
  xEvent.strokeStyle = '#1a7a30';
  xEvent.lineWidth   = 1.5;
  xEvent.setLineDash([3, 4]);
  xEvent.beginPath();
  strPts.forEach(([t, sv], i) => {
    const x = tx(t), y = sy(sv);
    i === 0 ? xEvent.moveTo(x, y) : xEvent.lineTo(x, y);
  });
  xEvent.stroke();
  xEvent.setLineDash([]);
  xEvent.restore();

  // ── distance trace (bold, primary, on top) ──
  const pts2d = samps.map(s => [toSec(s[0]), s[1]]);
  // plotLine uses MARGIN.r for clipping — pass adjusted W so it uses our mr
  // Instead draw directly with the same sentinel logic:
  xEvent.save();
  xEvent.lineWidth   = 2.5;
  xEvent.strokeStyle = '#c05000';
  let penDown = false, lastValidY = null;
  const markers = [];
  xEvent.beginPath();
  pts2d.forEach(([t, d]) => {
    const x = tx(t);
    if (isSentinel(d)) {
      if (penDown) xEvent.stroke();
      penDown = false; xEvent.beginPath();
      if (lastValidY !== null) markers.push({x, y: lastValidY});
    } else {
      const y = ty(d);
      if (!penDown) { xEvent.moveTo(x, y); penDown = true; }
      else            xEvent.lineTo(x, y);
      lastValidY = y;
    }
  });
  if (penDown) xEvent.stroke();
  // × markers
  const R = 4;
  xEvent.strokeStyle = '#cc4400'; xEvent.lineWidth = 1.5;
  markers.forEach(({x, y}) => {
    xEvent.beginPath();
    xEvent.moveTo(x-R,y-R); xEvent.lineTo(x+R,y+R);
    xEvent.moveTo(x+R,y-R); xEvent.lineTo(x-R,y+R);
    xEvent.stroke();
  });
  xEvent.restore();

  // Left Y-axis label (distance)
  xEvent.save();
  xEvent.font = FONT_SM; xEvent.fillStyle = COL_TEXT;
  xEvent.textAlign = 'center'; xEvent.textBaseline = 'middle';
  xEvent.translate(10, mt + ph / 2);
  xEvent.rotate(-Math.PI / 2);
  xEvent.fillText('m', 0, 0);
  xEvent.restore();

  // Right Y-axis label (strength)
  xEvent.save();
  xEvent.font = FONT_SM; xEvent.fillStyle = '#1a7a30';
  xEvent.textAlign = 'center'; xEvent.textBaseline = 'middle';
  xEvent.translate(W - 10, mt + ph / 2);
  xEvent.rotate(Math.PI / 2);
  xEvent.fillText('strength', 0, 0);
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

// ── distance readout (mm.mm fixed-width digits) ───────────────────────────────
function setReadout(dist_mm, isEv) {
  // Clamp to 0–99990 mm, display as mm.mm metres (2 integer, 2 fractional)
  const m      = Math.max(0, Math.min(99990, Math.round(dist_mm)));
  const meters = m / 1000;
  // padStart(5) gives "03.46" for 3456 mm, "40.00" for 40000 mm
  const str = meters.toFixed(2).padStart(5, '0');
  document.getElementById('r0').textContent = str[0];
  document.getElementById('r1').textContent = str[1];
  // str[2] is the dot — skip
  document.getElementById('r2').textContent = str[3];
  document.getElementById('r3').textContent = str[4];
  document.getElementById('readout').className = isEv ? 'ev' : '';
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
    fetch('/event')
      .then(r => r.json())
      .then(data => {
        evData = data;
        const t  = new Date(data.id * 1000);
        const hms = t.toTimeString().slice(0, 8);
        const ms  = t.getMilliseconds();
        const ts  = hms + '.' + Math.floor(ms / 100);
        document.getElementById('lbl-event').innerHTML =
          '<b>' + ts + '</b> · ' + (data.category || '?') +
          ' · ' + data.duration_ms + ' ms' +
          ' · <b style="color:#1a2530;font-weight:bold">' + (data.object_range_mm / 1000).toFixed(3) + ' m</b>';
        drawEvent();
      });
  }

  dot.className = 'live';   // CSS @keyframes handles 1 Hz blink; set once, stays
};

src.onerror = () => { dot.className = ''; };
</script>
</body>
</html>"""

@_flask_app.route('/')
def _index():
    return Response(_PAGE_HTML, mimetype='text/html')

@_flask_app.route('/event')
def _event():
    with _web_lock:
        ev = _last_event
    if ev is None:
        return Response('{}', mimetype='application/json')
    return Response(json.dumps(ev), mimetype='application/json')

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
# ---------------------------------------------------------------------------
n_seed = BASELINE_SECS * SAMPLE_RATE
print(f"Collecting {n_seed} samples ({BASELINE_SECS}s) to seed baseline...")
seed_dist = []
last_baseline_print = 0.0
while len(seed_dist) < n_seed:
    d, s, temp_c = read_sample()
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
    times = np.array([s[0] for s in samples], dtype=float)
    dists = np.array([s[1] for s in samples], dtype=float)

    # --- Clean dropouts: replace sentinels with linearly interpolated values ---
    dropout_mask = np.isin(dists, list(DROPOUT_VALUES))
    if dropout_mask.any():
        dists[dropout_mask] = np.nan
        nans = np.isnan(dists)
        idx  = np.arange(len(dists))
        dists[nans] = np.interp(idx[nans], idx[~nans], dists[~nans])

    excursion = bg_mean - dists          # positive when object present
    max_exc   = excursion.max()
    if max_exc <= 0:
        return None                      # no real event

    # --- Rise-time ratio classifier ---
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
    below_mask = excursion > 0
    if category == "vehicle":
        # Median of samples on the flat floor (within 10% of minimum dist)
        min_dist     = dists[below_mask].min()
        floor_mask   = below_mask & (dists <= min_dist + 0.10 * max_exc)
        object_range = float(np.median(dists[floor_mask]))
    else:
        # Parabolic fit over 9 points centred on raw minimum; analytic vertex
        raw_min_idx = int(np.argmin(dists))
        half        = 4
        lo          = max(0, raw_min_idx - half)
        hi          = min(len(dists) - 1, raw_min_idx + half)
        window_t    = times[lo:hi+1]
        window_d    = dists[lo:hi+1]
        if len(window_d) >= 3:
            coeffs      = np.polyfit(window_t, window_d, 2)
            a, b, _     = coeffs
            if a > 0:                    # parabola opens up → has minimum
                t_vertex     = -b / (2 * a)
                object_range = float(np.polyval(coeffs, t_vertex))
            else:
                object_range = float(dists[raw_min_idx])
        else:
            object_range = float(dists[raw_min_idx])

    # --- Duration: time spent with excursion > 50% of nominal excursion ---
    nominal_exc = bg_mean - object_range
    dur_mask    = excursion > 0.50 * nominal_exc
    dur_ms      = int(times[dur_mask][-1] - times[dur_mask][0]) if dur_mask.any() else 0

    return {
        "category":        category,
        "object_range_mm": round(object_range),
        "duration_ms":     dur_ms,
        "rise_ratio":      round(rise_ratio, 3),
    }

def save_event(start_epoch, samples, result):
    fname = os.path.join(OUTPUT_DIR, f"event_{start_epoch:.3f}.csv")
    with open(fname, 'w') as f:
        hms_string = datetime.fromtimestamp(start_epoch).strftime('%H:%M:%S.%f')[:-3]
        f.write(f"# start={hms_string} epoch={start_epoch:.3f} bg_mean={bg_mean:.2f} bg_std={bg_std:.3f} units=mm\n")
        if result:
            f.write(f"# category={result['category']}  object_range={result['object_range_mm']}mm"
                    f"  duration={result['duration_ms']}ms  rise_ratio={result['rise_ratio']}\n")
        f.write("offset_ms,dist_mm,strength\n")
        for offset_ms, dist, strength in samples:
            f.write(f"{offset_ms},{dist},{strength}\n")
    print(f"  saved {fname}  ({len(samples)} samples)")
    if result:
        print(f"  category={result['category']}  object_range={result['object_range_mm']}mm"
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
            with _web_lock:
                _web_state['category'] = result['category'] if result else ''
                _web_state['last_event_id'] = event_start_epoch
                _last_event = {
                    "id":             event_start_epoch,
                    "category":       result['category']        if result else '',
                    "object_range_mm":result['object_range_mm'] if result else 0,
                    "duration_ms":    result['duration_ms']     if result else 0,
                    "bg_mean":        bg_mean,
                    "trigger_ms":     trigger_ms,
                    # Full 100-sps samples: [offset_ms, dist_mm, strength]
                    "samples":        [[s[0], s[1], s[2]] for s in event_samples],
                }
            in_event = False
            below_count = 0
            pre_buf.clear()
