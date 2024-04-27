from psyaitools.func.IOs.base64_xfccs import *
import soundfile as sf

# aaa = base64_xfccs('../../comDat/testBase64.csv', 33, '../../comDat/test.webm', feature_set="mfcc")
# aaa = wav_xfccs('../../comDat/test.wav')

fs, sig = getWavDat('../../comDat/test.wav')
print(type(sig))
print(type(fs))
aaa = getLoud('../../comDat/test.wav')
