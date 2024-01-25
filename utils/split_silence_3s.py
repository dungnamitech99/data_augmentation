import os
from glob import glob
import soundfile as sf


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


if __name__ == "__main__":
    silence_3s = glob("/tf/train_hiFPToi/ftel5_5Dec2023/silence/*.wav")
    for silence_path in silence_3s:
        name = silence_path.split("/")[-1].split(".")[-2]
        x, sr = audioread(silence_path)
        count = 1
        for i in range(0, len(x), 48000):
            if len(x[i : i + 48000]) == 48000:
                audiowrite(
                    x[i : i + 48000],
                    sr,
                    f"/tf/train_hiFPToi/ftel5_5Dec2023/silence_3s/{name}_{str(count)}.wav",
                )
                count += 1
