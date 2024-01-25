import os
import yaml
import models as wuw_models
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util

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

    wav_data = tf.placeholder(tf.float32, [160000, 1], name="decoded_sample_data")
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

    # fingerprint_frequency_size = 10
    # fingerprint_time_size = 64
    # mfcc_ = tf.reshape(mfcc_, [-1, fingerprint_frequency_size * fingerprint_time_size])

    # mfcc_ = tf.expand_dims(mfcc_, axis=0)

    nodes = [op.name for op in tf.get_default_graph().get_operations()]
    print(nodes)
    print(nodes[-1])

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [nodes[-1]]
    )
    pb_path = os.path.join("/tf/train_hiFPToi/exp/lstm-add-ftel5", "preprocessing.pb")
    tf.train.write_graph(
        frozen_graph_def,
        "/tf/train_hiFPToi/exp/lstm-add-ftel5",
        "preprocessing.pb",
        as_text=False,
    )
    print(f"Saved frozen graph to {pb_path}")

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "/tf/train_hiFPToi/exp/lstm-add-ftel5/preprocessing.pb",
        input_arrays=[nodes[0]],
        output_arrays=[nodes[-1]],
    )
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open("/tf/train_hiFPToi/exp/lstm-add-ftel5/preprocessing.tflite", "wb") as f:
        f.write(tflite_model)
