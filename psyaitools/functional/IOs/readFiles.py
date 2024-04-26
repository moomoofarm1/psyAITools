import pandas as pd
from scipy.io.wavfile import read

def getRecDat(path):
    df1 = pd.read_csv(path)
    return df1

def getWavDat(fpath):
    fs, sig = read(fpath)
    return fs, sig