import yaml
import os
import sys
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
from utils import load_audio

# if __name__ == "__main__":
#     tf.disable_eager_execution()
#     tf.reset_default_graph()
#     sess = tf.InteractiveSession()
#     with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", 'r') as conf_fp:
#         conf_yaml=yaml.safe_load(conf_fp)
#     model = wuw_models.select_model(conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"])
#     model_settings = model.prepare_model_settings()
#     fingerprint_size = conf_yaml["model_conf"]['fingerprint_size']
#     label_count = conf_yaml["model_conf"]['label_count']
#     fingerprint_input = tf.placeholder(tf.float32, [None, fingerprint_size], name='fingerprint_input')
#     ground_truth_input = tf.placeholder(tf.float32, [None, label_count], name='groundtruth_input')
#     logits, dropout_prob = model.forward(fingerprint_input, conf_yaml["model_conf"]["model_size_info"], is_training=False)
#     model.load_variables_from_checkpoint(sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_ds4a_1"))

#     # preprocessing graph
#     wav_filename_placeholder_ = tf.placeholder(tf.float32, [20800, 1], name="audio")
#     audio = wav_filename_placeholder_
#     # left_position_ = tf.placeholder(tf.int32, [], name="left_position")
#     # desired_audio_ = tf.slice(audio, (left_position_, 0), (20800, 1))
#     foreground_volume_placeholder_ = tf.placeholder(tf.float32, [], name="foreground_volume")
#     scaled_foreground = tf.multiply(audio, foreground_volume_placeholder_)
#     background_data_placeholder_ = tf.placeholder(tf.float32, [20800, 1], name="background_data")
#     background_volume_placeholder_ = tf.placeholder(tf.float32, [], name="background_volume")
#     background_mul = tf.multiply(background_data_placeholder_, background_volume_placeholder_)
#     background_add = tf.add(background_mul, scaled_foreground)
#     down_volume_placeholder_ = tf.placeholder(tf.float32, [],name="down_volume")
#     down_volume = tf.multiply(background_add, down_volume_placeholder_)
#     background_clamp = tf.clip_by_value(down_volume, -1.0, 1.0)
#     spectrogram_ = tf.raw_ops.AudioSpectrogram(input=background_clamp,
#                                                   window_size=model_settings['window_size_samples'],
#                                                   stride=model_settings['window_stride_samples'],
#                                                   magnitude_squared=True)
#     mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram_,
#                             sample_rate=model_settings['sample_rate'],
#                             dct_coefficient_count=conf_yaml["feat_conf"]['dct_coefficient_count'],
#                             upper_frequency_limit=conf_yaml["feat_conf"]['upper_frequency_limit'],
#                             lower_frequency_limit=conf_yaml["feat_conf"]['lower_frequency_limit'],
#                             filterbank_channel_count=conf_yaml["feat_conf"]['filterbank_channel_count'])


#     val_cem_mean = 0
#     total_accuracy = 0
#     total_conf_matrix = None

#     dic = {"positive": [],
#            "negative": []}
#     with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val10_shuffle.txt", "r") as f:
#         for line in f:
#             components = line.strip().split()
#             if components[0] == "positive":
#                 dic["positive"].append(components[-1])
#             else:
#                 dic["negative"].append(components[-1])

#     labels = []
#     predicts = []
#     for label in dic:
#         for path in tqdm(dic[label], total=len(dic[label])):
#             labels.append(label)
#             wav_input = load_audio(path, desired_samples=48000)
#             wav_input = wav_input.audio
#             print("---------------------------------------->wav_input: ", wav_input.shape)
#             print(type(wav_input))
#             print(wav_input)
#             wav_input = wav_input[11200:32000]
#             background_reshaped = np.zeros([20800, 1])
#             background_volume = 0
#             # left_position = 11200.0
#             down_volume_random = np.random.uniform(0, 1)
#             mfcc = sess.run(mfcc_, feed_dict={wav_filename_placeholder_: wav_input,
#                                             #    left_position_: left_position,
#                                                foreground_volume_placeholder_: 1,
#                                                background_data_placeholder_: background_reshaped,
#                                                background_volume_placeholder_: background_volume,
#                                                down_volume_placeholder_: 1})
#             mfcc = mfcc.flatten()
#             mfcc = np.expand_dims(mfcc, axis=0)
#             print(mfcc.shape)
#             output = sess.run([logits], feed_dict={fingerprint_input: mfcc})
#             print(output)

#             print(np.argmax(output[0]))
#             print(["positive", "negative"][np.argmax(output[0])])
#             predict = np.argmax(output[0])
#             predicts.append(["positive", "negative"][predict])

#     # print(labels)
#     # print(predicts)
#     print(len(labels))
#     print(len(predicts))
#     print(labels)
#     print(predicts)
#     print(classification_report(labels, predicts))
#     print(confusion_matrix(labels, predicts))
###############################################################################################################################
if __name__ == "__main__":
    tf.disable_eager_execution()
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", "r") as conf_fp:
        conf_yaml = yaml.safe_load(conf_fp)
    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    model_settings = model.prepare_model_settings()
    fingerprint_size = conf_yaml["model_conf"]["fingerprint_size"]
    label_count = conf_yaml["model_conf"]["label_count"]
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name="fingerprint_input"
    )
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name="groundtruth_input"
    )
    logits, dropout_prob = model.forward(
        fingerprint_input, conf_yaml["model_conf"]["model_size_info"], is_training=False
    )
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_ds4a_1")
    )

    # preprocessing graph
    wav_filename_placeholder_ = tf.placeholder(tf.float32, [20800, 1], name="audio")
    audio = wav_filename_placeholder_
    # left_position_ = tf.placeholder(tf.int32, [], name="left_position")
    # desired_audio_ = tf.slice(audio, (left_position_, 0), (20800, 1))
    # foreground_volume_placeholder_ = tf.placeholder(tf.float32, [], name="foreground_volume")
    # scaled_foreground = tf.multiply(audio, foreground_volume_placeholder_)
    # background_data_placeholder_ = tf.placeholder(tf.float32, [20800, 1], name="background_data")
    # background_volume_placeholder_ = tf.placeholder(tf.float32, [], name="background_volume")
    # background_mul = tf.multiply(background_data_placeholder_, background_volume_placeholder_)
    # background_add = tf.add(background_mul, scaled_foreground)
    # down_volume_placeholder_ = tf.placeholder(tf.float32, [],name="down_volume")
    # down_volume = tf.multiply(background_add, down_volume_placeholder_)
    # background_clamp = tf.clip_by_value(down_volume, -1.0, 1.0)
    spectrogram_ = tf.raw_ops.AudioSpectrogram(
        input=audio,
        window_size=model_settings["window_size_samples"],
        stride=model_settings["window_stride_samples"],
        magnitude_squared=True,
    )
    mfcc_ = tf.raw_ops.Mfcc(
        spectrogram=spectrogram_,
        sample_rate=model_settings["sample_rate"],
        dct_coefficient_count=conf_yaml["feat_conf"]["dct_coefficient_count"],
        upper_frequency_limit=conf_yaml["feat_conf"]["upper_frequency_limit"],
        lower_frequency_limit=conf_yaml["feat_conf"]["lower_frequency_limit"],
        filterbank_channel_count=conf_yaml["feat_conf"]["filterbank_channel_count"],
    )

    val_cem_mean = 0
    total_accuracy = 0
    total_conf_matrix = None

    dic = {"positive": [], "negative": []}
    with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val10_shuffle.txt", "r") as f:
        for line in f:
            components = line.strip().split()
            if components[0] == "positive":
                dic["positive"].append(components[-1])
            else:
                dic["negative"].append(components[-1])

    labels = []
    predicts = []
    for label in dic:
        for path in tqdm(dic[label], total=len(dic[label])):
            labels.append(label)
            wav_input = load_audio(path, desired_samples=48000)
            wav_input = wav_input.audio
            print(
                "---------------------------------------->wav_input: ", wav_input.shape
            )
            print(type(wav_input))
            print(wav_input)
            wav_input = wav_input[11200:32000]
            background_reshaped = np.zeros([20800, 1])
            background_volume = 0
            # left_position = 11200.0
            down_volume_random = np.random.uniform(0, 1)
            mfcc = sess.run(mfcc_, feed_dict={wav_filename_placeholder_: wav_input})
            mfcc = mfcc.flatten()
            print("mfcc: ", mfcc)
            mfcc = np.expand_dims(mfcc, axis=0)
            print(mfcc.shape)
            output = sess.run([logits], feed_dict={fingerprint_input: mfcc})
            print(output)

            print(np.argmax(output[0]))
            print(["positive", "negative"][np.argmax(output[0])])
            predict = np.argmax(output[0])
            predicts.append(["positive", "negative"][predict])

    # print(labels)
    # print(predicts)
    # print(len(labels))
    # print(len(predicts))
    # print(labels)
    # print(predicts)
    # print(classification_report(labels, predicts))
    # print(confusion_matrix(labels, predicts))
