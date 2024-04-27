# -*- coding: utf-8 -*-
# matlab
# https://superkogito.github.io/spafe/features/_features.html
from spafe.features import bfcc, mfcc
from spafe.features import gfcc, lfcc, cqcc, lpc, msrcc, ngcc, pncc, psrcc, rplp, spfeats

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
        The feature set to extract. 
        Recommended options are 'bfcc', and 'mfcc'. The default is 'mfcc'.
        Other options are 'gfcc', 'lfcc', 'cqcc', 'lpc', 'msrcc', 'ngcc', 'pncc', 'psrcc', 'rplp', 'spfeats'.
    
    Returns
    -------
    numpy array
        The extracted features.
    """
    # print("The feature set: ", feature_set)

    return {
        'bfcc': _bfcc,
        'cqcc': _cqcc,
        'gfcc': _gfcc,
        'lfcc': _lfcc,
        'lpc': _lpc,
        'msrcc' : _msrcc,
        'ngcc' : _ngcc,
        'pncc' : _pncc,
        'psrcc' : _psrcc,
        'rplp' : _rplp,
        'spfeats' : _spfeats,
    }.get(feature_set, _mfcc)(sig, fs)

def _bfcc(sig, fs):
    return bfcc.bark_spectrogram(sig, fs)

def _cqcc(sig, fs):
    return cqcc.cqt_spectrogram(sig, fs)

def _gfcc(sig, fs):
    return gfcc.erb_spectrogram(sig, fs)

def _lfcc(sig, fs):
    return lfcc.linear_spectrogram(sig, fs)

def _lpc(sig, fs):
    return lpc.lpc(sig, fs)

def _mfcc(sig, fs):
    return mfcc.mel_spectrogram(sig, fs)    

def _msrcc(sig, fs):
    return msrcc.msrcc(sig, fs)

def _ngcc(sig, fs):
    return ngcc.ngcc(sig, fs)

def _pncc(sig, fs):
    return pncc.medium_time_power_calculation(sig, fs)

def _psrcc(sig, fs):
    return psrcc.psrcc(sig, fs)

def _rplp(sig, fs):
    return rplp.plp(sig, fs)

def _spfeats(sig, fs):
    return spfeats.extract_feats(sig, fs) 