#!/usr/bin/env python3
"""
build_lidar_distance_table.py (standalone) - Batch-run vehicle event
analysis across all lidar event CSV files, to build a lookup table of
camera-to-vehicle distance vs event time. This backfills distance data for
the pre-EXIF-SubjectDistance archive (before ~2026-06-16).

Standalone: does not import plot_lidar_event.py (avoids its matplotlib
and zoneinfo dependencies, neither of which is available/needed on
headless Python 3.7 machines like rp49). Only requires numpy + pandas.

camera_to_vehicle_m = CAMERA_TO_LIDAR_OFFSET_M + object_range_mm/1000

Output CSV columns:
  event_csv_path, event_start_epoch, event_start_local, category,
  object_range_mm, camera_to_vehicle_m, duration_ms, rise_ratio

Usage:
    python3 build_lidar_distance_table.py /mnt/bluecherry/LIDAR1 lidar_distances.csv
"""

import sys
import glob
import os
import re
import time
import numpy as np
import pandas as pd

CAMERA_TO_LIDAR_OFFSET_M = 11.0
SENTINEL = 40000  # values >= this are "no return" flags (TF02-Pro uses 45000)
RISE_TIME_THRESHOLD = 0.35  # >= this -> pedestrian, < this -> vehicle
DROPOUT_VALUES = {0, 45000}  # mm: 0=overload, 45000=loss-of-signal


def load_event(path):
    """Parse one event_*.csv file: '# key=value' header comments plus an
    offset_ms,dist_mm,strength data table. Identical logic to
    plot_lidar_event.py's load_event(), reproduced here standalone."""
    meta = {}
    meta_str = {}  # string-valued fields, e.g. category=vehicle
    offset_ms, dist_mm, strength = [], [], []
    with open(path, newline='') as f:
        for line in f:
            if line.startswith('#'):
                # Numeric fields: key=123.45 (also handles "9864mm" by
                # capturing the leading number and ignoring the unit suffix)
                for m in re.finditer(r'(\w+)=([\d.]+)\s*(?:mm|ms)?', line):
                    meta[m.group(1)] = float(m.group(2))
                # String fields: key=word (e.g. category=vehicle). Only
                # keep ones not already captured numerically above.
                for m in re.finditer(r'(\w+)=(\w+)', line):
                    key, val = m.group(1), m.group(2)
                    if key not in meta:
                        meta_str[key] = val
                continue
            row = line.strip()
            if not row:
                continue
            parts = row.split(',')
            if len(parts) != 3:
                continue
            # Generic header-row detection: a real data row's first field
            # is always numeric (either "offset_ms" in ms or "sample" as a
            # row index, depending on file format/era). Skip any row whose
            # first field doesn't parse as a number, rather than matching
            # one specific expected header string -- older files use a
            # "sample,dist_mm,strength" header instead of
            # "offset_ms,dist_mm,strength".
            try:
                first_val = float(parts[0])
            except ValueError:
                continue
            offset_ms.append(first_val)
            dist_mm.append(float(parts[1]))
            strength.append(float(parts[2]))

    # Older files use "epoch=" instead of "event_start_epoch=" in the
    # comment header. Normalize so downstream code only needs to look for
    # one key.
    if 'event_start_epoch' not in meta and 'epoch' in meta:
        meta['event_start_epoch'] = meta['epoch']

    meta['_str'] = meta_str  # stash string fields for analyze_event to use

    data = {
        'offset_ms': np.array(offset_ms),
        'dist_mm':   np.array(dist_mm),
        'strength':  np.array(strength),
    }
    return meta, data


def analyze_event(meta, data):
    """
    If the file's own comment header already includes precomputed
    category/object_range/duration/rise_ratio (some older captures have
    this, written at logging time), prefer those values directly rather
    than recomputing -- recomputation depends on the first data column
    being true elapsed milliseconds, but some older files use a plain
    row-sample-index instead, which would silently produce wrong
    duration_ms/rise_ratio (object_range_mm is taken directly from the
    file in this case too, sidestepping the issue entirely rather than
    relying on it coming out correct anyway).

    Otherwise falls back to full recomputation (logic ported from
    plot_lidar_event.py's analyze_event()).
    """
    meta_str = meta.get('_str', {})
    precomputed_category = meta_str.get('category')
    if precomputed_category is not None and 'object_range' in meta:
        # File already has its own precomputed analysis -- use it directly
        # rather than recomputing against a possibly-unreliable time axis.
        return {
            "category": precomputed_category,
            "object_range_mm": round(meta['object_range']),
            "duration_ms": int(meta.get('duration', 0)),
            "rise_ratio": round(meta.get('rise_ratio', float('nan')), 3),
        }

    bg_mean = meta.get('bg_mean')
    if bg_mean is None:
        return None

    times = data['offset_ms'].astype(float)
    dists = data['dist_mm'].astype(float)

    if len(dists) == 0:
        # Header present but no data rows -- truncated/empty event file
        # (seen in some older captures). .max() on an empty array raises
        # ValueError, so this must be checked before any reduction.
        return None

    dropout_mask = np.isin(dists, list(DROPOUT_VALUES))
    if dropout_mask.any():
        dists = dists.copy()
        dists[dropout_mask] = np.nan
        nans = np.isnan(dists)
        idx = np.arange(len(dists))
        if nans.all():
            # Every sample was a dropout sentinel -- nothing to interpolate from
            return None
        dists[nans] = np.interp(idx[nans], idx[~nans], dists[~nans])

    excursion = bg_mean - dists
    max_exc = excursion.max()
    if max_exc <= 0:
        return None

    above_lo = np.where(excursion >= 0.10 * max_exc)[0]
    above_hi = np.where(excursion >= 0.90 * max_exc)[0]
    if len(above_lo) == 0 or len(above_hi) == 0:
        rise_ratio = 1.0
    else:
        lead_time = times[above_hi[0]] - times[above_lo[0]]
        trail_time = times[above_lo[-1]] - times[above_hi[-1]]
        total_time = times[above_lo[-1]] - times[above_lo[0]]
        rise_ratio = float((lead_time + trail_time) / total_time) if total_time > 0 else 1.0

    category = "pedestrian" if rise_ratio >= RISE_TIME_THRESHOLD else "vehicle"

    below_mask = excursion > 0
    if category == "vehicle":
        min_dist = dists[below_mask].min()
        floor_mask = below_mask & (dists <= min_dist + 0.10 * max_exc)
        object_range = float(np.median(dists[floor_mask]))
    else:
        raw_min_idx = int(np.argmin(dists))
        half = 4
        lo = max(0, raw_min_idx - half)
        hi = min(len(dists) - 1, raw_min_idx + half)
        window_t = times[lo:hi + 1]
        window_d = dists[lo:hi + 1]
        if len(window_d) >= 3:
            coeffs = np.polyfit(window_t, window_d, 2)
            a, b, _ = coeffs
            if a > 0:
                t_vertex = -b / (2 * a)
                object_range = float(np.polyval(coeffs, t_vertex))
            else:
                object_range = float(dists[raw_min_idx])
        else:
            object_range = float(dists[raw_min_idx])

    nominal_exc = bg_mean - object_range
    dur_mask = excursion > 0.50 * nominal_exc
    if dur_mask.any():
        dur_ms = int(times[dur_mask][-1] - times[dur_mask][0])
    else:
        dur_ms = 0

    return {
        "category": category,
        "object_range_mm": round(object_range),
        "duration_ms": dur_ms,
        "rise_ratio": round(rise_ratio, 3),
    }


def epoch_to_local_str(epoch):
    """
    Dependency-free local-time label using only the stdlib time module
    (no zoneinfo/pytz needed). Uses the system's configured local
    timezone via time.localtime(), so it's correct as long as the
    machine itself is set to America/Los_Angeles (or whatever local tz
    applies) -- standard for a Raspberry Pi running with local tz
    configured via raspi-config / /etc/timezone. DST is handled
    correctly by the OS in this approach, unlike a hardcoded UTC offset.
    """
    if epoch is None:
        return ''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/bluecherry/LIDAR1'
    out_csv = sys.argv[2] if len(sys.argv) > 2 else 'lidar_distances.csv'

    pattern = os.path.join(root_dir, '**', 'event_*.csv')
    paths = sorted(glob.glob(pattern, recursive=True))
    print("Found {} event CSV files under {}".format(len(paths), root_dir))

    records = []
    n_failed = 0
    n_no_bg = 0
    n_pedestrian = 0
    n_no_epoch = 0
    failure_examples = []  # (path, exception_repr) for the first few failures

    for i, path in enumerate(paths):
        try:
            meta, data = load_event(path)
            result = analyze_event(meta, data)
        except Exception as e:
            n_failed += 1
            if len(failure_examples) < 10:
                failure_examples.append((path, repr(e)))
            continue

        if result is None:
            n_no_bg += 1
            continue

        if result['category'] != 'vehicle':
            n_pedestrian += 1
            continue

        epoch = meta.get('event_start_epoch')
        if epoch is None:
            # Older-format files may lack this field entirely. Skip rather
            # than crash; these can be backfilled separately once we know
            # what alternate timestamp source is available for them
            # (filename, file mtime, etc).
            n_no_epoch += 1
            continue

        camera_to_vehicle_m = CAMERA_TO_LIDAR_OFFSET_M + result['object_range_mm'] / 1000.0

        records.append({
            'event_csv_path': path,
            'event_start_epoch': epoch,
            'event_start_local': epoch_to_local_str(epoch),
            'category': result['category'],
            'object_range_mm': result['object_range_mm'],
            'camera_to_vehicle_m': camera_to_vehicle_m,
            'duration_ms': result['duration_ms'],
            'rise_ratio': result['rise_ratio'],
        })

        if (i + 1) % 500 == 0:
            print("  {}/{} processed...".format(i + 1, len(paths)))

    out = pd.DataFrame(records)
    if len(out) == 0:
        print("\nNo vehicle events with a usable event_start_epoch were found.")
        print("  Pedestrian events skipped: {}".format(n_pedestrian))
        print("  No bg_mean (unanalyzable): {}".format(n_no_bg))
        print("  No event_start_epoch found: {}".format(n_no_epoch))
        print("  Failed to parse: {}".format(n_failed))
        if failure_examples:
            print("\n  First {} failures (path, exception):".format(len(failure_examples)))
            for p, err in failure_examples:
                print("    {}".format(p))
                print("      {}".format(err))
        out.to_csv(out_csv, index=False)
        print("Saved (empty) to {}".format(out_csv))
        return

    out = out.sort_values('event_start_epoch').reset_index(drop=True)
    out.to_csv(out_csv, index=False)

    print("\nDone.")
    print("  Vehicle events written: {}".format(len(out)))
    print("  Pedestrian events skipped: {}".format(n_pedestrian))
    print("  No bg_mean (unanalyzable): {}".format(n_no_bg))
    print("  No event_start_epoch found: {}".format(n_no_epoch))
    print("  Failed to parse: {}".format(n_failed))
    if failure_examples:
        print("\n  First {} failures (path, exception):".format(len(failure_examples)))
        for p, err in failure_examples:
            print("    {}".format(p))
            print("      {}".format(err))
    print("Saved to {}".format(out_csv))

    print("\nDistance distribution (camera_to_vehicle_m):")
    print(out['camera_to_vehicle_m'].describe())
    print("\nDate range: {} to {}".format(
        out['event_start_local'].min(), out['event_start_local'].max()))


if __name__ == '__main__':
    main()
