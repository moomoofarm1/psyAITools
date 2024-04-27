# -*- coding: utf-8 -*-
import pyloudnorm as pyln

def loudness(fs, sig):
    meter = pyln.Meter(fs)  # create BS.1770 meter
    return meter.integrated_loudness(sig)