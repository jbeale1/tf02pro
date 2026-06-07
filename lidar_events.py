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
import serial.tools.list_ports   # enable port scan for correct device


# --- Configuration ---
VERSION = "1.007  2026-06-07"  # switch classifier to rise-time ratio
OUTPUT_DIR    = 'events'

SAMPLE_RATE   = 100          # nominal samples/sec
BASELINE_SECS = 10           # seconds of quiet data to seed the baseline
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
            in_event = False
            below_count = 0
            pre_buf.clear()
