import yaml
import os
import sys
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import soundfile as sf
import models as wuw_models
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.framework import graph_util
from utils import get_checkpoint_path, load_audio
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
    mark_sample = 2 * 16000
    input_len = int(1.3 * 16000)
    tf.disable_eager_execution()
    sess = tf.InteractiveSession()

    # backgrounds = []
    # with open("data/backgound/all_bg_list.txt", "r") as f:
    #     files = f.readlines()
    # for file in files:
    #     audio, sr = audioread(file.strip())
    #     backgrounds.append(audio)

    with open("/tf/train_hiFPToi/conf/lstm-s-3.yaml", "r") as conf_fp:
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
    scaled_foreground = tf.multiply(wav_data, 1)

    background_data_placeholder_ = tf.compat.v1.placeholder(
        tf.float32, [model_settings["desired_samples"], 1], name="background_data"
    )
    background_volume_placeholder_ = tf.compat.v1.placeholder(
        tf.float32, [], name="background_volume"
    )
    background_mul = tf.multiply(
        background_data_placeholder_, background_volume_placeholder_
    )
    background_add = tf.add(background_mul, scaled_foreground)

    down_volume_placeholder_ = tf.compat.v1.placeholder(
        tf.float32, [], name="down_volume"
    )
    down_volume = tf.multiply(background_add, down_volume_placeholder_)
    background_clamp = tf.clip_by_value(down_volume, -1.0, 1.0)

    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=wav_data,
        window_size=model_settings["window_size_samples"],
        stride=model_settings["window_stride_samples"],
        magnitude_squared=True,
    )

    mfcc_ = tf.raw_ops.Mfcc(
        spectrogram=spectrogram,
        sample_rate=model_settings["sample_rate"],
        dct_coefficient_count=conf_yaml["feat_conf"]["dct_coefficient_count"],
        upper_frequency_limit=conf_yaml["feat_conf"]["upper_frequency_limit"],
        lower_frequency_limit=conf_yaml["feat_conf"]["lower_frequency_limit"],
        filterbank_channel_count=conf_yaml["feat_conf"]["filterbank_channel_count"],
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
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/lstm_ds4a_4_v1")
    )

    # get audio path
    dic = {"positive": [], "negative": []}
    with open(
        "/tf/train_hiFPToi/data/trainsets/wuw_ds4a/test15_mixed_negative_add_positive_volumn_normed_shuffle.txt",
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

    background_reshaped = np.zeros([model_settings["desired_samples"], 1])
    background_volume = 0
    result = {}
    for label in dic:
        for path in tqdm(dic[label], total=len(dic[label])):
            print(path)
            # if "audio_beam8_phase1_fa1/" not in path:
            print(path)
            labels.append(label)
            print(path)
            # wav_data, samplerate = audioread(path)
            # input_data = np.expand_dims(wav_data, 1)
            # input_data = input_data.astype(np.float32)
            # input_data = input_data[mark_sample-input_len:mark_sample, :]
            # input_data = np.float32(input_data)
            # logits = sess.run([logits], feed_dict={wav_data: input_data})
            # print("--------------------------------------------------")sao
            # x, sr = audioread(path)
            x = load_audio(path, desired_samples=48000)
            # print("x: ", x)
            x = x.audio
            # print(x.shape)
            input_data = x[mark_sample - input_len : mark_sample, :]
            input_data = np.float32(input_data)
            # print(input_data.shape)
            down_volume_random = np.random.uniform(0, 1)
            if down_volume_random < 0.1:
                down_volume_placeholder = np.random.uniform(0.9, 1)
            else:
                down_volume_placeholder = 1

            output = sess.run(
                [logits],
                feed_dict={
                    wav_data: input_data,
                    down_volume_placeholder_: down_volume_placeholder,
                    background_data_placeholder_: background_reshaped,
                    background_volume_placeholder_: background_volume,
                },
            )
            # print(output[0])
            # print(np.argmax(output[0]))
            # print(["positive", "negative"][np.argmax(output[0])])
            predict = np.argmax(output[0])
            with open("/tf/train_hiFPToi/analysis_results/false_alarm.txt", "a") as f:
                # print("predict: ", ["positive", "negative"][predict])
                # print("label: ", label)
                if (["positive", "negative"][predict] == "positive") and (
                    label == "negative"
                ):
                    f.write(path + "\n")

            with open("/tf/train_hiFPToi/analysis_results/miss_rate.txt", "a") as f:
                if (["positive", "negative"][predict] == "negative") and (
                    label == "positive"
                ):
                    f.write(path + "\n")

            predicts.append(["positive", "negative"][predict])
            # print("predicts: ", predicts)
            result[path] = list(output[0][0].astype(float))

    with open(
        "/tf/train_hiFPToi/analysis_results/result_latest_original_lstm_4_v1.json", "w"
    ) as f:
        json.dump(result, f)

    print(len(labels))
    print(len(predicts))
    print(classification_report(labels, predicts))
    print(confusion_matrix(labels, predicts))
