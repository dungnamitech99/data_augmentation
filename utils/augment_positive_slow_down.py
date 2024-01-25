import os
import soundfile as sf
from glob import glob
import subprocess
import numpy as np


def audioread(path, norm=False, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print("WARNING: Audio type not supported")

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x**2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        x = x.sum(axis=0) / x.shape[0]
        if norm:
            rms = (x**2).mean() ** 0.5
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
    subprocess.call(
        f'ffmpeg -i {file_in} -filter:a "atempo={rate}" {file_out}', shell=True
    )


if __name__ == "__main__":
    positive_paths = []
    audio_files = glob("Allb_3s/*.wav") + glob("ftel5_5Dec2023/positive_3s/*.wav")

    for positive_path in audio_files:
        if "volumn_normed" not in positive_path:
            positive_paths.append(positive_path)

    audio_names = []
    ratio = 0.9
    # ratio = np.random.uniform(0.9, 1)
    # ratio = np.random.uniform(0.8, 1)
    word_end_time = 2
    for audio_file in positive_paths:
        audio_name = audio_file.split("/")[-1]
        if audio_name not in audio_names:
            change_speed(audio_file, f"ftel5_positive_slow_down/{audio_name}", ratio)
            audio_names.append(audio_name)

    slow_down_audios = glob("ftel5_positive_slow_down/*.wav")
    for slow_down_audio in slow_down_audios:
        x, sr = audioread(slow_down_audio)
        audio_name = slow_down_audio.split("/")[-1]
        cut_samples_front = round((word_end_time - word_end_time / ratio) * sr)
        # print(cut_samples_front)
        x = x[-cut_samples_front:]
        cut_samples_back = len(x) - 3 * 16000
        x = x[:-cut_samples_back]
        # print(len(x))
        audiowrite(x, sr, f"ftel5_positive_slow_down/{audio_name}")
