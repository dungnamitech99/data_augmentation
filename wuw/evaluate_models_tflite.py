import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix


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


if __name__ == "__main__":
    # initialize model
    model = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn_v1_5/model.tflite"
    )
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # get audio path
    dic = {"positive": [], "negative": []}
    with open(
        "/tf/train_hiFPToi/data/trainsets/trainset4_ftel4_balanced/label_time_wav_test_15.txt",
        "r",
    ) as f:
        for line in f:
            components = line.strip().split()
            if components[0] == "positive":
                dic["positive"].append(components[-1])
            else:
                dic["negative"].append(components[-1])

    # evaluate
    labels = []
    predicts = []
    mark_sample = 2 * 16000
    shift_offset = int(0.01 * 16000)
    input_len = int(1.3 * 16000)

    for label in dic:
        for path in tqdm(dic[label], total=len(dic[label])):
            labels.append(label)
            wav_data, samplerate = audioread(path)
            input_data = np.expand_dims(wav_data, 1)
            input_data = input_data.astype(np.float32)
            input_data = input_data[mark_sample - input_len : mark_sample, :]
            input_data = np.float32(input_data)
            model.set_tensor(input_details[0]["index"], input_data)
            model.invoke()
            output_data = model.get_tensor(output_details[0]["index"])
            predicts.append(["positive", "negative"][np.argmax(output_data)])

    print(len(labels))
    print(len(predicts))
    print(classification_report(labels, predicts))
    print(confusion_matrix(labels, predicts))
