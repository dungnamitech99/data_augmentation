import os
import soundfile as sf
from glob import glob
import subprocess
import numpy as np


def audioread(path, norm = False, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0)/x.shape[0]
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr

def audiowrite(data, fs, destpath):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    
    sf.write(destpath, data, fs)
    return


def change_speed(file_in, file_out, rate):
    subprocess.call(f'ffmpeg -i {file_in} -filter:a "atempo={rate}" {file_out}', shell=True)

if __name__ == "__main__":
    audio_files = glob("positive_3s/**/*.wav", recursive=True) \
                 + glob("positive123_3s/ftel123_clone_aligned/**/*.wav", recursive=True)
    
    audio_names = []
    for audio_file in audio_files:
        audio_name = audio_file.split("/")[-1]
        if audio_name not in audio_names:
            change_speed(audio_file, f"augment_positive_slow_down/{audio_name}", 0.9)
            audio_names.append(audio_name)

    slow_down_audios = glob("augment_positive_slow_down/*.wav")
    for slow_down_audio in slow_down_audios:
        x, sr = audioread(slow_down_audio)
        audio_name = slow_down_audio.split("/")[-1]
        cut_samples = len(x) - 3 * 16000
        # print(cut_samples)
        x = x[cut_samples:]
        # print(len(x)/sr)
        audiowrite(x, sr, f"augment_positive_slow_down/{audio_name}")

        