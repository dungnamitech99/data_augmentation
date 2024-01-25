import os
import math
import random
import numpy as np
from enum import Enum
import tensorflow as tf
from tqdm import tqdm
from utils import load_audio
import soundfile as sf


def audiowrite(data, fs, destpath, norm=False):
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, data, fs)
    return


class Label(Enum):
    POSITIVE = 0
    NEGATIVE = 1
    SILENCE = 2


class AudioLoader:
    def __init__(self, data_conf, feat_conf, model_conf, cache=False):
        self.cache = cache
        self.data_conf = data_conf
        self.feat_conf = feat_conf
        self.silence_percentage = data_conf["silence_percentage"]
        self.train_path = data_conf["train_index"]
        self.val_path = data_conf["val_index"]
        self.test_path = data_conf["test_index"]
        self.background_path = data_conf["background_index"]
        self.model_settings = model_conf

        self.data_index = {"validation": [], "testing": [], "training": []}
        self.background_data = []

        self.prepare_data_index()
        self.prepare_background_data()
        self.mfcc_input_, self.mfcc_ = self.prepare_processing_graph()

    def prepare_data_index(self):
        for index in ["validation", "testing", "training"]:
            if index == "validation":
                input_path = self.val_path
            elif index == "testing":
                input_path = self.test_path
            elif index == "training":
                input_path = self.train_path
            else:
                raise Exception("File does not exists")

            utt3s_desired_samples = self.model_settings["sample_rate"] * 3
            with open(input_path) as input_file:
                for line in input_file:
                    coms = line.strip().split()
                    word = coms[0]
                    right_position = float(coms[1]) * self.model_settings["sample_rate"]
                    path = coms[2]

                    if word == "positive":
                        self.data_index[index].append(
                            {
                                "label": Label.POSITIVE,
                                "file": path,
                                "right_position": right_position,
                                "wav_data": None,
                            }
                        )
                    elif word == "negative":
                        self.data_index[index].append(
                            {
                                "label": Label.NEGATIVE,
                                "file": path,
                                "right_position": right_position,
                                "wav_data": None,
                            }
                        )
                    elif word == "silence":
                        self.data_index[index].append(
                            {
                                "label": Label.NEGATIVE,
                                "file": path,
                                "right_position": right_position,
                                "wav_data": None,
                            }
                        )

        # Make sure the ordering is random.
        print("-----")
        total_count_label = {}
        for set_index in ["validation", "testing", "training"]:
            random.shuffle(self.data_index[set_index])

            # count label
            count_label = {}
            samples = self.data_index[set_index]
            for sample in samples:
                label = sample["label"]
                if label in count_label:
                    count_label[label] += 1
                else:
                    count_label[label] = 1
                if label in total_count_label:
                    total_count_label[label] += 1
                else:
                    total_count_label[label] = 1
            print(
                f"Set index: {set_index} - Count: {count_label.items()} - Sum:",
                sum(count_label.values()),
            )
        print(
            f"Total count: {total_count_label.items()} - Sum: {sum(total_count_label.values())}"
        )
        print("-----")

    def prepare_background_data(self):
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
            wav_loader = tf.io.read_file(wav_filename_placeholder)
            wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
            try:
                with open(self.background_path) as input_file:
                    print("Load background")
                    for line in input_file:
                        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        # wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: line.strip()}).audio
                        # print(wav_data.shape)
                        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        wav_data = sess.run(
                            wav_decoder,
                            feed_dict={wav_filename_placeholder: line.strip()},
                        ).audio.flatten()
                        # print("-----------------------------------------------------")
                        # print(wav_data)
                        # print(wav_data.shape)
                        # print(type(wav_data))
                        # print("-----------------------------------------------------")
                        if wav_data.shape[0] > self.model_settings["desired_samples"]:
                            self.background_data.append(wav_data)
            except Exception as e:
                pass
                # raise Exception('No background wav files were found')
        # if not self.background_data:
        #     pass
        # raise Exception('No background wav files were found')

    def prepare_processing_graph(self):
        desired_samples = self.model_settings["desired_samples"]
        utt3s_desired_samples = self.model_settings["sample_rate"] * 3

        if self.cache:
            wav_filename_placeholder_ = tf.compat.v1.placeholder(
                tf.float32, [utt3s_desired_samples, 1], name="audio"
            )
            audio = wav_filename_placeholder_
        else:
            wav_filename_placeholder_ = tf.compat.v1.placeholder(
                tf.string, [], name="wav_filename"
            )
            wav_loader = tf.io.read_file(wav_filename_placeholder_)
            audio, _ = tf.audio.decode_wav(
                wav_loader,
                desired_channels=1,
                desired_samples=utt3s_desired_samples,
                name="wav_input",
            )

        left_position_ = tf.compat.v1.placeholder(tf.int32, [], name="left_position")
        desired_audio_ = tf.slice(audio, (left_position_, 0), (desired_samples, 1))

        # print("wav_decoder",audio.shape)
        # print("left_position_=",left_position_)

        # Allow the audio sample's volume to be adjusted.
        foreground_volume_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [], name="foreground_volume"
        )
        scaled_foreground = tf.multiply(desired_audio_, foreground_volume_placeholder_)

        # print("scaled_foreground",scaled_foreground.shape)

        # Mix in background noise.
        background_data_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [desired_samples, 1], name="background_data"
        )
        background_volume_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [], name="background_volume"
        )
        background_mul = tf.multiply(
            background_data_placeholder_, background_volume_placeholder_
        )
        background_add = tf.add(background_mul, scaled_foreground)

        # Down volume
        down_volume_placeholder_ = tf.compat.v1.placeholder(
            tf.float32, [], name="down_volume"
        )
        down_volume = tf.multiply(background_add, down_volume_placeholder_)
        background_clamp = tf.clip_by_value(down_volume, -1.0, 1.0)

        # print("background_clamp",background_clamp.shape)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram_ = tf.raw_ops.AudioSpectrogram(
            input=background_clamp,
            window_size=self.model_settings["window_size_samples"],
            stride=self.model_settings["window_stride_samples"],
            magnitude_squared=True,
        )
        # print("spectrogram_:",spectrogram_.shape)
        mfcc_ = tf.raw_ops.Mfcc(
            spectrogram=spectrogram_,
            sample_rate=self.model_settings["sample_rate"],
            dct_coefficient_count=self.feat_conf["dct_coefficient_count"],
            upper_frequency_limit=self.feat_conf["upper_frequency_limit"],
            lower_frequency_limit=self.feat_conf["lower_frequency_limit"],
            filterbank_channel_count=self.feat_conf["filterbank_channel_count"],
        )

        # print("mfcc_",mfcc_)
        # fmean, fvar = tf.nn.moments(mfcc_, axes=[1], keepdims=True, name="mfcc_moments")
        # fstd = tf.math.sqrt(fvar)
        # zeromean_mfcc_ = mfcc_ - fmean
        # normed_mfcc_ = tf.math.divide_no_nan(zeromean_mfcc_, fstd, name='normed_feats')

        return {
            "wav_filename_placeholder_": wav_filename_placeholder_,
            "foreground_volume_placeholder_": foreground_volume_placeholder_,
            "left_position_": left_position_,
            "background_data_placeholder_": background_data_placeholder_,
            "background_volume_placeholder_": background_volume_placeholder_,
            "down_volume_placeholder_": down_volume_placeholder_,
        }, mfcc_

    def load_batch(
        self,
        sess,
        batch_size=100,
        offset=0,
        background_frequency=0,
        background_volume_range=0,
        background_silence_frequency=0,
        background_silence_volume_range=0,
        down_volume_frequency=0,
        down_volume_range=0,
        time_shift=0,
        augment_pos_to_nev=None,
        augment_mask=None,
        mode="training",
    ):
        # Pick one of the partitions to choose samples from.

        candidates = self.data_index[mode]
        if batch_size == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(batch_size, len(candidates) - offset))

        # Data and labels will be populated and returned.
        # print("-------------------------------------------")
        # print(batch_size)
        # print(self.model_settings['fingerprint_size'])
        # print(sample_count)
        # print("-------------------------------------------")
        data = np.zeros((sample_count, self.model_settings["fingerprint_size"]))
        labels = np.zeros((sample_count, self.model_settings["label_count"]))
        desired_samples = self.model_settings["desired_samples"]
        # print("desired_sample: ", desired_samples)
        utt3s_desired_samples = self.model_settings["sample_rate"] * 3
        # print("utt3s_desired_samples: ", utt3s_desired_samples)
        use_background = self.background_data and (mode == "training")
        # print("---------------------------------------------")
        # print("offset", offset)
        # print("---------------------------------------------")
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in range(offset, offset + sample_count):
            # Pick which audio sample to use.
            sample = candidates[i]
            # print("file: ", sample["file"])

            # If we're time shifting, set up the offset for this sample.
            # print("time_shift: ", time_shift)
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0

            sample_label = sample["label"]
            # print("----------------------------------------------")
            # print(augment_pos_to_nev)
            # print("----------------------------------------------")
            # print("augment_pos_to_nev: ", augment_pos_to_nev)
            if augment_pos_to_nev and sample_label == Label.POSITIVE:
                # print("accept augment_pos_to_nev")
                prob_random = np.random.uniform(0, 1)
                # print("--------------------------------------------")
                # print("prob_random", prob_random)
                # print("--------------------------------------------")
                if prob_random < augment_pos_to_nev["freq"]:
                    sample_label = Label.NEGATIVE
                    if np.random.randint(0, 2) == 0:
                        # shift left 0.5s
                        time_shift_amount -= int(
                            0.5 * self.model_settings["sample_rate"]
                        )
                    else:
                        # shift right 0.8s
                        time_shift_amount += int(
                            0.8 * self.model_settings["sample_rate"]
                        )

            # print("time_shift_amount: ", time_shift_amount)
            left_position = (
                sample["right_position"] - desired_samples + time_shift_amount
            )
            # print("left_position: ", left_position)
            if left_position < 0:
                left_position = 0
                # print("-------------------------------hi-------------------------------")
            if left_position >= utt3s_desired_samples - desired_samples:
                left_position = utt3s_desired_samples - desired_samples - 1
                # print("-------------------------------he-------------------------------")

            if self.cache:
                if sample["wav_data"] is None:
                    sample["wav_data"] = load_audio(
                        sample["file"], utt3s_desired_samples
                    )
                wav_input = sample["wav_data"].audio
            else:
                wav_input = sample["file"]

            # print("---------------------------------------->wav_input: ", wav_input.shape)
            # print(type(wav_input))
            # print(wav_input)

            input_dict = {
                self.mfcc_input_["wav_filename_placeholder_"]: wav_input,
                self.mfcc_input_["left_position_"]: left_position,
            }

            # Choose a section of background noise to mix in.
            # print("------------------use background--------------------------", use_background)
            if use_background:
                # print("------------------desired_samples-------------------------", self.model_settings['desired_samples'])
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) == 48000:
                    background_clipped = background_samples
                else:
                    background_offset = np.random.randint(
                        0,
                        len(background_samples)
                        - self.model_settings["desired_samples"],
                    )
                    background_clipped = background_samples[
                        background_offset : (background_offset + desired_samples)
                    ]
                # print("---------------------------------------")
                # print("background_clipped", background_clipped)
                # print("background_clipped", background_clipped.shape)
                # print("---------------------------------------")
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                # print("---------------------------------------")
                # print("background_reshaped", background_reshaped.shape)
                # print("---------------------------------------")
                background_random = np.random.uniform(0, 1)
                if (
                    sample_label == Label.SILENCE
                    and background_random < background_silence_frequency
                ):
                    background_volume = np.random.uniform(
                        0, background_silence_volume_range
                    )
                elif (
                    sample_label != Label.SILENCE
                    and background_random < background_frequency
                ):
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                # print("---------------no background------------------")
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            # print("------------------------background_volume------------------------", background_volume)
            input_dict[
                self.mfcc_input_["background_data_placeholder_"]
            ] = background_reshaped
            input_dict[
                self.mfcc_input_["background_volume_placeholder_"]
            ] = background_volume

            # If we want silence, mute out the main sample but leave the background.
            if sample_label == Label.SILENCE:
                input_dict[self.mfcc_input_["foreground_volume_placeholder_"]] = 0
            else:
                # print('foreground_volume_placeholder_', 1)
                input_dict[
                    self.mfcc_input_["foreground_volume_placeholder_"]
                ] = np.random.uniform(0.4, 1)

            # Down volume
            down_volume_random = np.random.uniform(0, 1)
            # print("down_volume_frequency: ", down_volume_frequency)
            if down_volume_random < down_volume_frequency:
                input_dict[
                    self.mfcc_input_["down_volume_placeholder_"]
                ] = np.random.uniform(down_volume_range, 1)
            else:
                # print("down_volume_placeholder_", 1)
                input_dict[self.mfcc_input_["down_volume_placeholder_"]] = 1

                # Run the graph to produce the output audio.
            mfcc_i = sess.run(self.mfcc_, feed_dict=input_dict)
            # result = sess.run([self.mfcc_, self.background_clamp_], feed_dict=input_dict)
            # audio = result[1]
            # print(len(audio)/16000)
            # audiowrite(audio, 16000, f"audio_test/{str(i)}.wav")
            # print("-----------------------------------------")
            # print("11111111111111111111111111111111111111111")
            # print("mfcc_i", type(mfcc_i))
            # print("mfcc_i", mfcc_i.shape)
            # print("-----------------------------------------")
            # print("-----------------augment mask-------------------", augment_mask)
            if augment_mask:
                # print(mfcc_i.shape)
                _, frames_count, coefs_count = mfcc_i.shape
                # print("frames_count", frames_count)
                # print("coefs_count", coefs_count)
                prob_random = np.random.uniform(0, 1)
                # print("prob_random", prob_random)
                if prob_random < augment_mask["freq_freq"]:
                    feat0 = np.random.randint(
                        0, coefs_count - augment_mask["freq_param"]
                    )
                    count0 = np.random.randint(0, augment_mask["freq_param"])
                    mfcc_i[:, :, feat0 : feat0 + count0] = 0
                prob_random = np.random.uniform(0, 1)
                if prob_random < augment_mask["time_freq"]:
                    time0 = np.random.randint(
                        0, frames_count - augment_mask["time_param"]
                    )
                    count0 = np.random.randint(0, augment_mask["time_param"])
                    mfcc_i[:, time0 : time0 + count0, :] = 0

            # print("--------------------------------------")
            # print("mfcc_i", type(mfcc_i))
            # print("mfcc_i", mfcc_i.shape)
            # print("mfcc_i.flatten", mfcc_i.flatten().shape)
            # print("data shape", data.shape)
            # print("data", data[0].shape)
            # print("label", labels.shape)
            # print("--------------------------------------")
            data[i - offset, :] = mfcc_i.flatten()

            if sample_label == Label.POSITIVE:
                label_index = 0
            else:
                label_index = 1
            labels[i - offset, label_index] = 1
            # print("--------------------------------------")
            # print("label", labels)
            # print("--------------------------------------")
        return data, labels

    def size(self, mode="training"):
        return len(self.data_index[mode])

    def shuffle(self, set_index="training"):
        random.shuffle(self.data_index[set_index])
