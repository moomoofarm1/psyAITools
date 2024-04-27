from psyaitools.func.IOs.base64_xfccs import *
import numpy as np
from scipy.io.wavfile import read

# aaa = base64_xfccs('../../comDat/testBase64.csv', 33, '../../comDat/test.webm', feature_set="mfcc")
# aaa = wav_xfccs('../../comDat/test.wav')

def getWavDat(fpath, filetype="wav"):
    fs, sig = read(fpath)
    sig = sig / np.iinfo(np.int16).max
    return fs, sig
def getLoud(path):
    fs, sig = getWavDat(path)
    sig = sig / np.iinfo(np.int16).max
    return loudness(fs, sig)
fs, sig = getWavDat('../../comDat/test.wav')

print(type(sig.dtype))
# print(type(fs))

import pyloudnorm as pyln

def loudness(fs, sig):
    meter = pyln.Meter(fs)  # create BS.1770 meter
    return meter.integrated_loudness(sig)
aaa = loudness(fs, sig)
print(aaa)