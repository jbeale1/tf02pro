#!/usr/bin/env python

# TF02-Pro LIDAR event logger
# Detects objects crossing the beam and saves per-event CSVs.
# J.Beale  5-Jun-2026

import serial
import numpy as np
import time
import collections
import os

# --- Configuration ---
PORT          = '/dev/ttyUSB1'
OUTPUT_DIR    = 'events'

SAMPLE_RATE   = 100          # nominal samples/sec
BASELINE_SECS = 5            # seconds of quiet data to seed the baseline
TRIGGER_N     = 3            # consecutive samples below threshold to trigger
TRIGGER_SIGMA = 3.0          # sigma below baseline mean to trigger
REARM_SIGMA   = 3.0          # sigma — distance must return within this to re-arm
REARM_SECS    = 0.5          # seconds within baseline to consider event over
MAX_EVENT_SECS = 3.0         # hard cap on event recording length

PRE_TRIGGER   = 10           # samples to prepend from rolling buffer
EMA_ALPHA     = 0.001        # EMA coefficient for slow baseline drift (~1000-sample TC)

# Quiet-check threshold: only update baseline when per-batch std is low.
# From data, quiet individual-sample std ≈ 1–2 cm.  We gate on a live
# estimate, but seed it from the initial baseline period.
QUIET_STD_MULT = 4.0         # baseline update gated to: std < QUIET_STD_MULT * bg_std

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
ser = serial.Serial(PORT, 115200, timeout=0.1)

def read_sample():
    """Block until one valid 9-byte TF02-Pro frame is parsed. Returns (dist, strength, quality)."""
    while True:
        time.sleep(0.005) # avoid busy loop; TF02-Pro sends at ~100 Hz, so this is a short wait
        if ser.in_waiting >= 9:
            if ser.read() == b'Y':
                if ser.read() == b'Y':
                    dist_l = ord(ser.read())
                    dist_h = ord(ser.read())
                    str_l  = ord(ser.read())
                    str_h  = ord(ser.read())
                    _      = ser.read()   # reserved
                    qual   = ord(ser.read())
                    _      = ser.read()   # checksum (ignored)
                    dist     = dist_h * 256 + dist_l
                    strength = str_h  * 256 + str_l
                    return dist, strength, qual

# ---------------------------------------------------------------------------
# Phase 1: seed baseline from BASELINE_SECS seconds of quiet readings
# ---------------------------------------------------------------------------
n_seed = BASELINE_SECS * SAMPLE_RATE
print(f"Collecting {n_seed} samples ({BASELINE_SECS}s) to seed baseline...")
seed_dist = []
while len(seed_dist) < n_seed:
    d, s, q = read_sample()
    seed_dist.append(d)

bg_mean = float(np.mean(seed_dist))
bg_std  = float(np.std(seed_dist))
print(f"Baseline: mean={bg_mean:.2f} cm  std={bg_std:.3f} cm")
print(f"Trigger threshold: < {bg_mean - TRIGGER_SIGMA * bg_std:.2f} cm")
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

def save_event(start_epoch, samples):
    fname = os.path.join(OUTPUT_DIR, f"event_{start_epoch:.3f}.csv")
    with open(fname, 'w') as f:
        f.write(f"# event_start_epoch={start_epoch:.3f}  bg_mean={bg_mean:.2f}  bg_std={bg_std:.3f}\n")
        f.write("offset_ms,dist_cm,strength\n")
        for offset_ms, dist, strength in samples:
            f.write(f"{offset_ms},{dist},{strength}\n")
    print(f"  saved {fname}  ({len(samples)} samples)")

sample_idx = 0   # global sample counter used for ms offsets within events

while True:
    d, strength, qual = read_sample()
    t_now = time.time()
    sample_idx += 1

    trigger_thresh = bg_mean - TRIGGER_SIGMA * bg_std

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
            print(f"EVENT triggered at {t_now:.3f}  dist={d}  threshold={trigger_thresh:.1f}")

        else:
            # Update slow EMA baseline only when signal is quiet
            if bg_std > 0 and abs(d - bg_mean) < QUIET_STD_MULT * bg_std:
                bg_mean = EMA_ALPHA * d + (1 - EMA_ALPHA) * bg_mean
                # Update bg_std as EMA of squared deviation
                bg_std = float(np.sqrt(
                    EMA_ALPHA * (d - bg_mean) ** 2 + (1 - EMA_ALPHA) * bg_std ** 2
                ))

    else:
        # Recording event
        t_pre0 = pre_buf[0][0] if pre_buf else t_now
        offset_ms = round((t_now - event_start_epoch) * 1000)
        event_samples.append((offset_ms, d, strength))

        # Check re-arm condition: sustained return to baseline
        if abs(d - bg_mean) <= REARM_SIGMA * bg_std:
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
            save_event(event_start_epoch, event_samples)
            in_event = False
            below_count = 0
            pre_buf.clear()
