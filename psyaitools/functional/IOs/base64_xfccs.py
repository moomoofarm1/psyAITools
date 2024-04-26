# -*- coding: utf-8 -*-
import numpy as np
from .readFiles import *
from .base64_wav import *
from ..representation.xfccs import *

def base64_xfccs(path, audioRow, fileName,feature_set="mfcc"):
    dat1 = getRecDat(path)
    base64ToWebm(dat1.loc[audioRow,"response"], path, method=2)
    webm2Wav(fileName)
    fs, sig = getWavDat(fileName + '.wav')
    out1, _ = xfccs(fs, sig, feature_set=feature_set)
    return np.mean(out1, axis=1)