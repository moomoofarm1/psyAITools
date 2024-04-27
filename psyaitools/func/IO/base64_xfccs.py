# -*- coding: utf-8 -*-
import numpy as np
from .readFiles import *
from .base64_wav import *
from ..representation.xfccs import *
from ..representation.loud import *

def base64_xfccs(path, audioRow, fileName,feature_set="mfcc"):
    dat1 = getRecDat(path)
    base64ToWebm(dat1.loc[audioRow, "response"], fileName, method=2)
    webm2Wav(fileName)
    # Split the filename into name and extension
    nam1, _ = fileName.rsplit('.', 1)
    fs, sig = getWavDat(nam1 + ".wav")
    out1, _ = xfccs(fs, sig, feature_set=feature_set)
    return np.mean(out1, axis=1)

def wav_xfccs(path, feature_set="mfcc"):
    fs, sig = getWavDat(path)
    out1, _ = xfccs(fs, sig, feature_set=feature_set)
    return np.mean(out1, axis=1)

def getLoud(path):
    fs, sig = getWavDat(path)
    sig = sig / np.iinfo(np.int16).max # loudness expects float64, not int16.
    return loudness(fs, sig)