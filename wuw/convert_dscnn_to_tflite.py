import os
import yaml
import models as wuw_models
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
from utils import get_checkpoint_path, load_audio


if __name__ == "__main__":
    tf.disable_eager_execution()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", "r") as conf_fp:
        conf_yaml = yaml.safe_load(conf_fp)

    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    model_settings = model.prepare_model_settings()

    wav_data = tf.placeholder(tf.float32, [20800, 1], name="decoded_sample_data")
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

    fmean, fvar = tf.nn.moments(mfcc_, axes=[1], keepdims=True, name="mfcc_moments")
    fstd = tf.math.sqrt(fvar)
    zeromean_mfcc_ = mfcc_ - fmean
    normed_mfcc_ = tf.math.divide_no_nan(zeromean_mfcc_, fstd, name="normed_feats")

    fingerprint_frequency_size = 10
    fingerprint_time_size = 64
    mfcc_ = tf.reshape(
        normed_mfcc_, [-1, fingerprint_frequency_size * fingerprint_time_size]
    )

    logits, dropout_prob = model.forward(
        mfcc_, conf_yaml["model_conf"]["model_size_info"], is_training=False
    )
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn")
    )
    tf.sigmoid(logits, name="labels_sigmoid")
    nodes = [op.name for op in tf.compat.v1.get_default_graph().get_operations()]

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["labels_sigmoid"]
    )
    pb_path = os.path.join(
        "/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn", "model.pb"
    )
    tf.compat.v1.train.write_graph(
        frozen_graph_def,
        os.path.dirname(pb_path),
        os.path.basename(pb_path),
        as_text=False,
    )
    print(f"Saved frozen graph to {pb_path}")

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays=["decoded_sample_data"], output_arrays=["labels_sigmoid"]
    )
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(
        "/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn/model_e2e.tflite", "wb"
    ) as f:
        f.write(tflite_model)
