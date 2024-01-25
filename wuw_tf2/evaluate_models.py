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
    # initialize preprocessing
    preprocessing = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn_v1_5/dscnn_fe.tflite"
    )
    preprocessing.allocate_tensors()
    input_details_preprocessing = preprocessing.get_input_details()
    output_details_preprocessing = preprocessing.get_output_details()

    # initialize model
    model = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn_v1_5/dscnn_core.tflite"
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
            frames = []
            sub_wav_end_pre = None
            sub_predicts = []
            for i in range(5):
                if i == 0:
                    end_sample = mark_sample - shift_offset
                else:
                    end_sample = sub_wav_end_pre + shift_offset
                assert end_sample in [31840, 32000, 32160, 32320, 32480]
                sub_wav = input_data[end_sample - input_len : end_sample, :]
                sub_wav = np.float32(sub_wav)
                frames.append(sub_wav)
                sub_wav_end_pre = end_sample

            for frame in frames:
                preprocessing.set_tensor(input_details_preprocessing[0]["index"], frame)
                preprocessing.invoke()
                output_data = preprocessing.get_tensor(
                    output_details_preprocessing[0]["index"]
                )
                model.set_tensor(input_details[0]["index"], output_data)
                model.invoke()
                output_data = model.get_tensor(output_details[0]["index"])
                sub_predicts.append(np.argmax(output_data[0]))

            counter = Counter(sub_predicts)

            if len(counter) > 1:
                if counter[0] > counter[1]:
                    predicts.append("positive")
                else:
                    predicts.append("negative")
            else:
                if any(list(counter.keys())):
                    predicts.append("negative")
                else:
                    predicts.append("positive")

    print(classification_report(labels, predicts))
    print(confusion_matrix(labels, predicts))
