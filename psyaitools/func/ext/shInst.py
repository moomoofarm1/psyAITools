# -*- coding: utf-8 -*-

#  TODO: Later to install ffmpeg, ffprobe, and ffplay.
# It should be like linux, windows, mac to decide which command to use.

import subprocess
import os, sys


ffmpeg1 = "echo 'Installing various tools and packages, including audio-video codecs, required for building FFmpeg && '" + \
"apt-get -y install autoconf automake build-essential libass-dev libfreetype6-dev libsdl1.2-dev libtheora-dev \
    libtool libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev pkg-config texinfo zlib1g-dev" 

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package]) # "pip", ""

# Check if ffmpeg is installed, no use.
def check_install_ffmpeg():
    try:
        ffmpeg_version = subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        print("ffmpeg is installed")
    except subprocess.CalledProcessError:
        print("ffmpeg is not installed")
        os.system(ffmpeg1)
        