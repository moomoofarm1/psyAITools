import pandas as pd
from scipy.io.wavfile import read

def getRecDat(path, filetype="csv"):
    df1 = pd.read_csv(path)
    return df1

def getWavDat(fpath, filetype="wav"):
    fs, sig = read(fpath + "." +filetype)
    return fs, sig