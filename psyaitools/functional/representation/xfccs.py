# -*- coding: utf-8 -*-
# matlab
# https://superkogito.github.io/spafe/features/_features.html
from spafe.features import bfcc, gfcc, lfcc, mfcc

def xfccs(fs, sig, feature_set="mfcc"):
    """
    Extracts the features from the audio signal.
    
    Parameters
    ----------
    fs : int
        The sampling frequency of the audio signal.
    sig : numpy array
        Audio signal from which to extract features.
    feature_set : str
        The feature set to extract. Options are 'bfcc', 'gfcc', 'lfcc', and 'mfcc'.
    
    Returns
    -------
    numpy array
        The extracted features.
    """
    # print("The feature set: ", feature_set)
    if feature_set == "bfcc":
        return bfcc.bark_spectrogram(sig, fs)
    elif feature_set == "gfcc":
        return gfcc.erb_spectrogram(sig, fs)
    elif feature_set == "lfcc":
        return lfcc.linear_spectrogram(sig, fs)
    elif feature_set == "mfcc":
        return mfcc.mel_spectrogram(sig, fs)
    else:
        raise ValueError("Invalid feature set. Options are 'bfcc', 'gfcc', 'lfcc', and 'mfcc'.")    