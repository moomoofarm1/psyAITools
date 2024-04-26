# -*- coding: utf-8 -*-
import numpy as np
from functional.IOs.base64_xfccs import *
np.random.seed(42)

path1 = '../../comDat/testBase64.csv'
path2 = '../../comDat/test'

def wavFeatures():
    print("Hello world!")
    
    # response
    #print(dat1.loc[24,"response"])
    #print(dat1.loc[26,"response"])
    #print(dat1.loc[30,"response"])
    #print(dat1.loc[33,"response"])

    dat_mel = base64_xfccs(path1, 33, path2, feature_set="mfcc")
    print(dat_mel.shape)

    print("Goodbye world!")

    return 0

if __name__ == "__main__":
    wavFeatures()