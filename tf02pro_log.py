#!/usr/bin/env python

# read range from Benewake TF02-Pro LIDAR module via serial
# J.Beale 5-Jun-2026

import serial
import numpy as np
import time

BATCH = 50
PORT = '/dev/ttyUSB1'  # adjust as needed   
ser = serial.Serial(PORT, 115200, timeout=0.1)

print("time,dist_mean,dist_std,dist_min,dist_max,strength_mean,strength_std,strength_min,strength_max,quality_mean,quality_min,quality_max")

while True:
    distances = []
    strengths = []
    qualities = []

    while len(distances) < BATCH:
        if ser.in_waiting >= 9:
            if ser.read() == b'Y' and ser.read() == b'Y':
                dist_l = ord(ser.read())
                dist_h = ord(ser.read())
                str_l  = ord(ser.read())
                str_h  = ord(ser.read())
                _      = ser.read()  # reserved
                qual   = ord(ser.read())
                _      = ser.read()  # checksum (ignored)

                distances.append(dist_h * 256 + dist_l)
                strengths.append(str_h * 256 + str_l)
                qualities.append(qual)

    d = np.array(distances)
    s = np.array(strengths)
    q = np.array(qualities)

    t = round(time.time(), 1)
    print(f"{t:.1f},{d.mean():.1f},{d.std():.2f},{d.min()},{d.max()},"
          f"{s.mean():.0f},{s.std():.1f},{s.min()},{s.max()},"
          f"{q.mean():.1f},{q.min()},{q.max()}")
