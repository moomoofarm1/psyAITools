import os, base64, audioread

def base64ToWebm(wavStr, fileName, method=2):
    
    assert isinstance(wavStr, str)
    assert isinstance(fileName, str)
    assert isinstance(method, int)
    
    if method == 1:
        missing_padding = len(wavStr) % 4
        if missing_padding != 0:
            wavStr += b'='* (4 - missing_padding)
        decode_string = base64.decodestring(wavStr)
    elif method == 2:
        lens = len(wavStr)
        lenx = lens - (lens % 4 if lens % 4 else 4)
        decode_string = base64.b64decode(wavStr[:lenx])
    elif method == 3:
        decode_string = base64.urlsafe_b64decode(wavStr)
    else:
        pass
    
    with open(fileName + ".webm", "wb") as wavFile:
        wavFile.write(decode_string)
    print("The file has been converted to .webm format.")
        
def webm2Wav(fileName, sampling_rate=None):
    
    assert isinstance(fileName, str)
    #assert isinstance(sampling_rate, int)
    
    channel = 1
   
    if os.path.exists(fileName + ".wav"): os.remove(fileName + ".wav")
    if os.path.isfile(fileName + ".webm"):
        with audioread.audio_open(fileName + ".webm") as f:
            sr = f.samplerate
            if (sampling_rate == None):
                sampling_rate = sr
                #print(f"The sampling rate: {sampling_rate}")
            else: pass
            command = "ffmpeg -loglevel quiet -i {} -ac {} -ar {} {}".format(
                fileName + ".webm",
                channel,
                sampling_rate,
                fileName + ".wav"
            )
            # print("The command: ", command)
            os.system(command)
    print("The file has been converted to .wav format.")
