import yaml
import os
import math
import sys
import numpy as np
from tqdm import tqdm
import soundfile as sf
import models as wuw_models
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.framework import graph_util
from utils import get_checkpoint_path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


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


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    mark_sample = 2 * 16000
    input_len = int(1.3 * 16000)
    tf.disable_eager_execution()
    sess = tf.InteractiveSession()

    with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", "r") as conf_fp:
        try:
            conf_yaml = yaml.safe_load(conf_fp)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(2)

    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    model_settings = model.prepare_model_settings()
    wav_data = tf.placeholder(
        tf.float32, [model_settings["desired_samples"], 1], name="decoded_sample_data"
    )
    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=wav_data,
        window_size=model_settings["window_size_samples"],
        stride=model_settings["window_stride_samples"],
        magnitude_squared=True,
    )
    mfcc_ = tf.raw_ops.Mfcc(
        spectrogram=spectrogram,
        sample_rate=model_settings["sample_rate"],
        dct_coefficient_count=model_settings["dct_coefficient_count"],
    )
    fingerprint_frequency_size = model_settings["dct_coefficient_count"]
    fingerprint_time_size = model_settings["spectrogram_length"]
    fingerprint_input = tf.reshape(
        mfcc_, [-1, fingerprint_time_size * fingerprint_frequency_size]
    )
    logits, _ = model.forward(
        fingerprint_input, model_settings["model_size_info"], is_training=False
    )
    tf.sigmoid(logits, name="labels_sigmoid")
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_ds4a_1")
    )

    # evaluate
    labels = []
    predicts = []
    mark_sample = 2 * 16000
    shift_offset = int(0.01 * 16000)
    input_len = int(1.3 * 16000)

    with open("/tf/train_hiFPToi/analysis_results/miss_rate.txt", "r") as f:
        paths = f.readlines()

    positive_scores = []
    negative_scores = []
    log_scores = []
    for path in tqdm(paths, total=len(paths)):
        labels.append("positive")
        x, sr = audioread(path.strip())
        x = np.expand_dims(x, 1)
        x = x.astype(np.float32)
        input_data = x[mark_sample - input_len : mark_sample, :]
        input_data = np.float32(input_data)
        output = sess.run([logits], feed_dict={wav_data: input_data})
        positive_score = sigmoid(output[0][0][0])
        negative_score = 1 - positive_score
        log_scores.append(np.log(positive_score / negative_score))
        # positive_scores.append(positive_score)
        # negative_scores.append(negative_score)
        # predict = np.argmax(output[0])
        # predicts.append(["positive", "negative"][predict])

    # print(len(labels))
    # print(len(predicts))
    # print(classification_report(labels, predicts))
    # print(confusion_matrix(labels, predicts))

    number_samples = range(1, 5264)
    # plt.figure(figsize=(40, 20))
    # plt.plot(number_samples[:100], positive_scores[:100], label="Positive scores")
    # plt.plot(number_samples[:100], negative_scores[:100], label="Negative scores")
    # plt.legend()
    # plt.savefig("/tf/train_hiFPToi/analysis_results/positive_scores.png")
    plt.plot(number_samples[:100], log_scores[:100])
    plt.savefig("/tf/train_hiFPToi/analysis_results/log_scores.png")
