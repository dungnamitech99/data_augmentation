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
    positive_3s = glob("/tf/train_hiFPToi/ftel5_5Dec2023/positive_3s/*.wav")
    silence_3s = glob("/tf/train_hiFPToi/ftel5_5Dec2023/silence_3s/*.wav")
    count_positive_smaller_3s = 0
    count_positive_greater_3s = 0
    count_negative_smaller_3s = 0
    count_negative_greater_3s = 0

    for positive_path in positive_3s:
        x, sr = audioread(positive_path)
        if len(x) < 48000:
            # print(len(x))
            count_positive_smaller_3s += 1

        if len(x) > 48000:
            # print(len(x))
            count_positive_greater_3s += 1
    print(count_positive_smaller_3s)
    print(count_positive_greater_3s)
    print("All positive samples are within 3s")

    for negative_path in silence_3s:
        x, sr = audioread(negative_path)
        if len(x) < 48000:
            # print(len(x))
            count_negative_smaller_3s += 1

        if len(x) > 48000:
            # print(len(x))
            count_negative_greater_3s += 1
    print(count_negative_smaller_3s)
    print(count_negative_greater_3s)
    print("All negative samples are within 3s")
