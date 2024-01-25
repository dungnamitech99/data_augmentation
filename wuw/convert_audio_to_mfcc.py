import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from glob import glob
from utils import load_audio
import pickle


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
        model_path="/tf/train_hiFPToi/exp/lstm-add-ftel5/preprocessing.tflite"
    )
    preprocessing.allocate_tensors()
    input_details_preprocessing = preprocessing.get_input_details()
    output_details_preprocessing = preprocessing.get_output_details()

    # initialize LSTM model
    lstm = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/lstm_ftel5full/lstm_ftel5full_core.tflite"
    )
    lstm.allocate_tensors()
    input_details_lstm = lstm.get_input_details()
    output_details_lstm = lstm.get_output_details()

    # initialize DSCNN model
    dscnn = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn/dscnn-add-ftel5-mfcc10-mvn_core.tflite"
    )
    dscnn.allocate_tensors()
    input_details_dscnn = dscnn.get_input_details()
    output_details_dscnn = dscnn.get_output_details()

    # get audio path
    dic_negative_audios = {}

    # record file paths
    # paths = glob("./nami_record/record_test/*.wav")
    with open("nami_record/testset_easy.txt", "r") as f:
        paths = f.readlines()
    paths = [path.strip().split(" ")[-1] for path in paths]

    for path in tqdm(paths, total=len(paths)):
        wav_input = load_audio(path, desired_samples=16000 * 10)
        wav_input = wav_input.audio

        # feed input wave to preprocessing
        preprocessing.set_tensor(input_details_preprocessing[0]["index"], wav_input)
        preprocessing.invoke()
        output_data = preprocessing.get_tensor(output_details_preprocessing[0]["index"])
        print(type(output_data))
        with open(
            "./nami_record/testset_easy_mfcc/" + path.split("/")[-1].replace(".wav", ".npy"), "wb"
        ) as f:
            np.save(f, output_data)

        num_mfcc = output_data.shape[1] * output_data.shape[2] // 640

        # LSTM model
        for i in range(num_mfcc):
            # print(i)
            mfcc = output_data[:, i : i + 64, :]
            # mfcc = mfcc.reshape(-1, 640)
            # print(mfcc.shape)
            # feed data to wuw models
            lstm.set_tensor(input_details_lstm[0]["index"], mfcc)
            lstm.invoke()
            score = lstm.get_tensor(output_details_lstm[0]["index"])
            # print(score)

            if path.split("/")[-1] not in dic_negative_audios:
                dic_negative_audios[path.split("/")[-1]] = [list(score[0])]
            else:
                dic_negative_audios[path.split("/")[-1]].append(list(score[0]))

        # DSCNN model
        for i in range(num_mfcc):
            mfcc = output_data[:, i : i + 64, :]
            # feed data to wuw models
            dscnn.set_tensor(input_details_dscnn[0]["index"], mfcc)
            dscnn.invoke()
            score = dscnn.get_tensor(output_details_dscnn[0]["index"])
            dic_negative_audios[path.split("/")[-1]][i] = dic_negative_audios[
                path.split("/")[-1]
            ][i] + list(score[0])

    # dump dict_negative_audios
    with open("./nami_record/dict_negative_audios_newlstm.pkl", "wb") as f:
        pickle.dump(dic_negative_audios, f)

    # with open("dict_negative_audios.pkl", "rb") as f:
    # test = pickle.load(f)

    # print(dic_negative_audios == test)
