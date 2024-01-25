import math
import sys
import ast
from data import *
from utils import *
import tf_slim as slim
import tensorflow.compat.v2 as tf
import tensorflow.compat.v1 as tf1
from layers import lstm, bc_resnet_blocks, sub_spectral_normalization


def select_model(data_conf, feat_conf, model_conf):
    name = model_conf["name"]
    if name == "lstm":
        return LSTM(data_conf, feat_conf, model_conf)
    elif name == "ds_cnn":
        return DS_CNN(data_conf, feat_conf, model_conf)
    elif name == "bc_resnet":
        return BC_RESNET(data_conf, feat_conf, model_conf)


class Model:
    def __init__(self, data_conf, feat_conf, model_conf):
        self.label_count = data_conf["label_count"]
        self.sample_rate = data_conf["sample_rate"]
        self.clip_duration_ms = data_conf["clip_duration_ms"]
        self.window_size_ms = feat_conf["window_size_ms"]
        self.window_stride_ms = feat_conf["window_stride_ms"]
        self.dct_coefficient_count = feat_conf["dct_coefficient_count"]
        self.upper_frequency_limit = feat_conf["upper_frequency_limit"]
        self.lower_frequency_limit = feat_conf["lower_frequency_limit"]
        self.filterbank_channel_count = feat_conf["filterbank_channel_count"]
        self.model_conf = model_conf

        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)
        self.window_size_samples = int(self.sample_rate * self.window_size_ms / 1000)
        self.window_stride_samples = int(
            self.sample_rate * self.window_stride_ms / 1000
        )
        self.length_minus_window = self.desired_samples - self.window_size_samples
        if self.length_minus_window < 0:
            self.spectrogram_length = 0
        else:
            self.spectrogram_length = 1 + int(
                self.length_minus_window / self.window_stride_samples
            )

        self.fingerprint_size = self.dct_coefficient_count * self.spectrogram_length

        self.model_conf["label_count"] = self.label_count
        self.model_conf["desired_samples"] = self.desired_samples
        self.model_conf["window_size_samples"] = self.window_size_samples
        self.model_conf["window_stride_samples"] = self.window_stride_samples
        self.model_conf["spectrogram_length"] = self.spectrogram_length
        self.model_conf["dct_coefficient_count"] = self.dct_coefficient_count
        self.model_conf["fingerprint_size"] = self.fingerprint_size
        self.model_conf["sample_rate"] = self.sample_rate

    def prepare_model_settings(self):
        return self.model_conf

    def forward(self, fingerprint_input, model_size_info=None, is_training=True):
        raise NotImplementedError

    @staticmethod
    def load_variables_from_checkpoint(sess, start_checkpoint):
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.restore(sess, start_checkpoint)


class LSTM(Model):
    def build(self, model_size_info=None, is_training=True):
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        num_classes = self.label_count
        projection_units = model_size_info[0]
        lstm_units = model_size_info[1]
        print("projection_units:", projection_units)
        print("lstm_units:", lstm_units)

        input_audio = tf.keras.layers.Input(
            shape=(
                input_time_size,
                input_frequency_size,
            )
        )
        net = input_audio
        net = lstm.LSTM(
            units=lstm_units,
            return_sequences=self.model_conf["return_sequences"],
            # stateful=self.model_conf["stateful"],
            use_peepholes=self.model_conf["use_peepholes"],
            num_proj=projection_units,
        )(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(
            units=self.model_conf["units1"], activation=self.model_conf["act1"]
        )(net)
        net = tf.keras.layers.Dense(units=num_classes)(net)
        if self.model_conf["return_softmax"]:
            net = tf.keras.layers.Activation("softmax")(net)
        return tf.keras.Model(input_audio, net)


class DS_CNN(Model):
    def build(self, model_size_info=None, is_training=True):
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        num_classes = self.label_count

        num_layers = model_size_info[0]
        conv_feat = [None] * num_layers
        conv_kt = [None] * num_layers
        conv_kf = [None] * num_layers
        conv_st = [None] * num_layers
        conv_sf = [None] * num_layers

        i = 1
        for layer_no in range(0, num_layers):
            conv_feat[layer_no] = model_size_info[i]
            i += 1
            conv_kt[layer_no] = model_size_info[i]
            i += 1
            conv_kf[layer_no] = model_size_info[i]
            i += 1
            conv_st[layer_no] = model_size_info[i]
            i += 1
            conv_sf[layer_no] = model_size_info[i]
            i += 1

        input_audio = tf.keras.layers.Input(
            shape=(
                input_time_size,
                input_frequency_size,
            )
        )
        # print("------------------------------------------")
        # print("input_audio: ", input_audio)
        # print("------------------------------------------")
        net = input_audio
        net = tf.keras.backend.expand_dims(net, axis=-1)
        for layer_no in range(0, num_layers):
            if layer_no == 0:
                net = tf.keras.layers.Conv2D(
                    kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                    dilation_rate=(1, 1),
                    filters=conv_feat[layer_no],
                    padding="same",
                    strides=(conv_st[layer_no], conv_sf[layer_no]),
                )(net)

                net = tf.keras.layers.Activation("relu")(net)

                net = tf.keras.layers.BatchNormalization(
                    momentum=0.98, center=1, scale=0, renorm=0
                )(net)

            else:
                net = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
                    dilation_rate=(1, 1),
                    padding="same",
                    strides=(conv_st[layer_no], conv_sf[layer_no]),
                )(net)

                net = tf.keras.layers.Activation("relu")(net)

                net = tf.keras.layers.BatchNormalization(
                    momentum=0.98, center=1, scale=0, renorm=0
                )(net)

                net = tf.keras.layers.Conv2D(
                    kernel_size=(1, 1), filters=conv_feat[layer_no]
                )(net)
                net = tf.keras.layers.Activation("relu")(net)
                net = tf.keras.layers.BatchNormalization(
                    momentum=0.98, center=1, scale=0, renorm=0
                )(net)

        net = tf.keras.layers.AveragePooling2D(
            pool_size=(int(net.shape[1]), int(net.shape[2]))
        )(net)

        net = tf.keras.layers.Flatten()(net)

        # net = tf.keras.layers.Dropout(rate=0.2)(net)
        net = tf.keras.layers.Dense(num_classes)(net)
        return tf.keras.Model(input_audio, net)


class BC_RESNET(Model):
    def build(self, is_training=True):
        input_frequency_size = self.dct_coefficient_count
        input_time_size = self.spectrogram_length
        num_classes = self.label_count

        dropouts = self.model_conf["dropouts"]
        filters = self.model_conf["filters"]
        blocks_n = self.model_conf["blocks_n"]
        strides = parse(self.model_conf["strides"])
        dilations = parse(self.model_conf["dilations"])
        pools = self.model_conf["pools"]
        padding = self.model_conf["paddings"]
        sub_groups = self.model_conf["sub_groups"]
        # print("--------------------------------------------")
        # print(dropouts)
        # print(type(dropouts))
        # print(filters)
        # print(type(filters))
        # print(blocks_n)
        # print(type(blocks_n))
        # print(strides)
        # print(type(strides))
        # print(dilations)
        # print(type(dilations))
        # print(pools)
        # print(type(pools))
        # print("--------------------------------------------")

        for l in (dropouts, filters, strides, dilations, pools):
            if len(blocks_n) != len(l):
                raise ValueError("all input lists have to be the same length")

        input_audio = tf.keras.layers.Input(
            shape=(
                input_time_size,
                input_frequency_size,
            )
        )
        net = input_audio
        net = tf.keras.backend.expand_dims(net)
        # print("-----------------------------------")
        # print("1")
        # print("net: ", net.shape)
        # print("-----------------------------------")

        if self.model_conf["paddings"] == "same":
            net = tf.keras.layers.Conv2D(
                filters=self.model_conf["first_filters"],
                kernel_size=5,
                strides=(1, 2),
                padding="same",
            )(net)
        else:
            net = tf.keras.layers.Conv2D(
                filters=self.model_conf["first_filters"],
                kernel_size=5,
                strides=(1, 2),
                padding="valid",
            )(net)
        # print("-----------------------------------")
        # print("2")
        # print("net: ", net.shape)
        # print(padding)
        # print("-----------------------------------")
        for n, n_filters, dilation, stride, dropout, pool in zip(
            blocks_n, filters, dilations, strides, dropouts, pools
        ):
            # print("-----------------------------------------------------")
            # print(n)
            # print(n_filters)
            # print(dilation)
            # print(stride)
            # print(dropout)
            # print(pool)
            # print("-----------------------------------------------------")
            net = bc_resnet_blocks.TransitionBlock(
                n_filters, dilation, stride, padding, dropout, sub_groups=sub_groups
            )(net)
            # print("-----------------------------------")
            # print("3")
            # print("net: ", net.shape)
            # print(padding)
            # print("-----------------------------------")
            for _ in range(n):
                net = bc_resnet_blocks.NormalBlock(
                    n_filters, dilation, 1, padding, dropout, sub_groups=sub_groups
                )(net)
                # print("-----------------------------------")
                # print("normal_block")
                # print("net: ", net.shape)
                # print(padding)
                # print("-----------------------------------")

            if pool > 1:
                if self.model_conf["max_pool"]:
                    net = tf.keras.layers.MaxPooling2D(
                        pool_size=(pool, 1), strides=(pool, 1)
                    )(net)
                else:
                    net = tf.keras.AveragePooling2D(
                        pool_size=(pool, 1), strides=(pool, 1)
                    )(net)

        if padding == "same":
            net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding="same")(net)
        else:
            net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding="valid")(net)

        # average out frequency dim
        net = tf.keras.backend.mean(net, axis=2, keepdims=True)

        net = tf.keras.layers.Conv2D(
            filters=self.model_conf["last_filters"], kernel_size=1, use_bias=False
        )(net)

        # average out time dim
        net = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(net)
        net = tf.keras.layers.Conv2D(
            filters=num_classes, kernel_size=1, use_bias=False
        )(net)

        # 1 ans 2 dims are equal to 1
        net = tf.squeeze(net, [1, 2])

        if self.model_conf["return_softmax"]:
            net = tf.keras.layers.Activation("softmax")(net)

        return tf.keras.Model(input_audio, net)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv[0], " <conf> <expdir>")
        sys.exit(1)

    conf_fn = sys.argv[1]
    expdir = sys.argv[2]
    logdir = expdir + "/logs"

    with open(conf_fn, "r") as conf_fp:
        try:
            conf_yaml = yaml.safe_load(conf_fp)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(2)

    conf_yaml["exp_conf"] = {"logdir": logdir, "expdir": expdir}

    augment_pos_to_nev = {"freq": conf_yaml["model_conf"]["augment_pos_to_nev_freq"]}
    augment_mask = {
        "freq_freq": conf_yaml["model_conf"]["augment_mask_freq_freq"],
        "freq_param": conf_yaml["model_conf"]["augment_mask_freq_param"],
        "time_freq": conf_yaml["model_conf"]["augment_mask_time_freq"],
        "time_param": conf_yaml["model_conf"]["augment_mask_time_param"],
    }

    # audio_loader = AudioLoader(conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"], cache=True)
    # train_fingerprints, train_ground_truth = audio_loader \
    #             .load_batch(batch_size=conf_yaml["model_conf"]["batch_size"],
    #                         offset=500,
    #                         background_frequency=conf_yaml["data_conf"]["background_frequency"],
    #                         background_volume_range=conf_yaml["data_conf"]["background_volume"],
    #                         background_silence_frequency=conf_yaml["data_conf"]["background_silence_frequency"],
    #                         background_silence_volume_range=conf_yaml["data_conf"]["background_silence_volume"],
    #                         time_shift=conf_yaml["data_conf"]["time_shift"],
    #                         down_volume_frequency=0.1,
    #                         down_volume_range=0.9,
    #                         augment_pos_to_nev=augment_pos_to_nev,
    #                         augment_mask=augment_mask,
    #                         mode='training')

    # print("-----------------------------------")
    # print(train_fingerprints.shape)
    # print(train_ground_truth.shape)
    # tf1.reset_default_graph()
    # config = tf1.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    # sess = tf1.Session(config=config)
    # tf1.keras.backend.set_session(sess)
    config = LSTM(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    model = config.build(conf_yaml["model_conf"]["model_size_info"])
    print(model.summary())
    # tf1.train.write_graph(sess.graph_def, conf_yaml["exp_conf"]["expdir"], "graph.pb")
    # model.forward(train_fingerprints, train_ground_truth)
    # training_steps_list = conf_yaml["model_conf"]["training_steps"]
    # learning_rates_list = conf_yaml["model_conf"]["learning_rate"]
    # train_size = audio_loader.size('training')
    # step = 0
    # epoch = 0
    # best_accuracy = 0
    # training_steps_max = np.sum(training_steps_list)
    # while step < training_steps_max + 1:
    #     epoch += 1
    #     audio_loader.shuffle(set_index='training')

    #     for offset in range(0, train_size, conf_yaml["model_conf"]["batch_size"]):
    #         step += 1
    #         if step >= training_steps_max + 1:
    #             break
    #         train_fingerprints, train_ground_truth = audio_loader \
    #                 .load_batch(batch_size=conf_yaml["model_conf"]["batch_size"],
    #                             offset=500,
    #                             background_frequency=conf_yaml["data_conf"]["background_frequency"],
    #                             background_volume_range=conf_yaml["data_conf"]["background_volume"],
    #                             background_silence_frequency=conf_yaml["data_conf"]["background_silence_frequency"],
    #                             background_silence_volume_range=conf_yaml["data_conf"]["background_silence_volume"],
    #                             time_shift=conf_yaml["data_conf"]["time_shift"],
    #                             down_volume_frequency=0.1,
    #                             down_volume_range=0.9,
    #                             augment_pos_to_nev=augment_pos_to_nev,
    #                             augment_mask=augment_mask,
    #                             mode='training')
    #         model.forward(train_fingerprints, train_ground_truth)
