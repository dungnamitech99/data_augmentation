import yaml
import os
import sys
import numpy as np
import soundfile as sf
import models as wuw_models
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.framework import graph_util
from utils import get_checkpoint_path


def audioread(path, norm=True, start=0, stop=None):
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
    tf.disable_eager_execution()
    sess = tf.InteractiveSession()

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
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/lstm_ds4a_1_v1")
    )
    nodes = [op.name for op in tf.get_default_graph().get_operations()]

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["labels_sigmoid"]
    )
    pb_path = os.path.join("/tf/train_hiFPToi/exp/lstm_ds4a_1_v1", "model.pb")
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(pb_path),
        os.path.basename(pb_path),
        as_text=False,
    )
    print(f"Saved frozen graph to {pb_path}")

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays=["decoded_sample_data"], output_arrays=["labels_sigmoid"]
    )
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(
        os.path.join("/tf/train_hiFPToi/exp/lstm_ds4a_1_v1", "model.tflite"), "wb"
    ) as f:
        f.write(tflite_model)

    # model = tf2.keras.models.load_model("/tf/train_hiFPToi/exp/test_bcresnet_2/model")
    # print(model.summary())

    # converter = tf2.lite.TFLiteConverter.from_saved_model("/tf/train_hiFPToi/exp/test_bcresnet_2/model")
    # converter.inference_type = tf.lite.constants.FLOAT
    # converter.experimental_new_quantizer = True
    # converter.experimental_enable_resource_variables = True
    # converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [
    #         tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.allow_custom_ops = True
    # converter.inference_input_type = tf.float32
    # converter.inference_output_type = tf.float32
    # tflite_model = converter.convert()
    # with open("/tf/train_hiFPToi/exp/test_bcresnet_2/bc_resnet.tflite", "wb") as f:
    #     f.write(tflite_model)

    # x, sr = audioread("/tf/train_hiFPToi/waves/train/ftel_train/positive_3s_isolate_clone_musicV30_16k_processed/wav/0a1c1122bb204c10a10348d41dc5beeb_1.96_3.84.wav")
    # input_data = np.expand_dims(x, 1)
    # input_data = input_data[11200:32000]
    # # print(input_data.shape)
    # # print(sr)
    # spectrogram = tf.raw_ops.AudioSpectrogram(input=input_data,
    #                                           window_size=640,
    #                                           stride=320,
    #                                           magnitude_squared=True)
    # mfcc_ = tf.raw_ops.Mfcc(spectrogram=spectrogram,
    #                         sample_rate=16000,
    #                         dct_coefficient_count=10)

    # fingerprint_frequency_size = 10
    # fingerprint_time_size = 64
    # fingerprint_input = tf.reshape(mfcc_, [-1, fingerprint_time_size, fingerprint_frequency_size])
    # logits = model.predict(fingerprint_input)
    # print(logits)
