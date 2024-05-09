from psyaitools.func.IO.base64_xfccs import *

# aaa = base64_xfccs('../../comDat/testBase64.csv', 33, '../../comDat/test.webm', feature_set="mfcc")
# aaa = wav_xfccs('../../comDat/test.wav')
aaa = getLoud('../../comDat/test.wav')
print(aaa)