from psyaitools.functional.IOs.base64_xfccs import base64_xfccs

aaa = base64_xfccs('../../comDat/testBase64.csv', 33, '../../comDat/test.wav', feature_set="mfcc")
aaa = wav_xfccs('../../comDat/test.wav')