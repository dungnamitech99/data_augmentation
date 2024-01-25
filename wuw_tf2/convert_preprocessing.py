import os
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util

if __name__ == "__main__":
    tf.disable_eager_execution()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    wav_data = tf.placeholder(tf.float32, [20800, 1], name="decoded_sample_data")
    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=wav_data, window_size=640, stride=320, magnitude_squared=True
    )
    mfcc_ = tf.raw_ops.Mfcc(
        spectrogram=spectrogram, sample_rate=16000, dct_coefficient_count=10
    )

    fingerprint_frequency_size = 10
    fingerprint_time_size = 64
    fingerprint_input = tf.reshape(
        mfcc_, [-1, fingerprint_time_size, fingerprint_frequency_size]
    )

    nodes = [op.name for op in tf.get_default_graph().get_operations()]
    # print(nodes)
    # print(nodes[-1])

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [nodes[-1]]
    )
    pb_path = os.path.join("/tf/train_hiFPToi/exp/test_bcresnet_2", "model.pb")
    tf.train.write_graph(
        frozen_graph_def,
        "/tf/train_hiFPToi/exp/test_bcresnet_2",
        "model.pb",
        as_text=False,
    )
    print(f"Saved frozen graph to {pb_path}")

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "/tf/train_hiFPToi/exp/test_bcresnet_2/model.pb",
        input_arrays=[nodes[0]],
        output_arrays=[nodes[-1]],
    )
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open("/tf/train_hiFPToi/exp/test_bcresnet_2/preprocessing.tflite", "wb") as f:
        f.write(tflite_model)
