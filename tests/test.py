from psyaitools.functional.IOs.base64_xfccs import base64_xfccs

aaa = base64_xfccs('../../comDat/testBase64.csv', 33, '../../comDat/test', feature_set="mfcc")
print(aaa.shape)