import tensorflow as tf


def model_split(pb_fn, fe_fn, core_fn, input_arrays, split_arrays, outout_arrays):
    converter_fe = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_fn, input_arrays=input_arrays, output_arrays=split_arrays
    )
    converter_fe.allow_custom_ops = True
    converter_fe.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_fe = converter_fe.convert()
    with open(fe_fn, "wb") as f:
        f.write(tflite_fe)

    converter_core_model = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        pb_fn, input_arrays=split_arrays, output_arrays=outout_arrays
    )
    converter_core_model.allow_custom_ops = True
    converter_core_model.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_core_model = converter_core_model.convert()
    with open(core_fn, "wb") as f:
        f.write(tflite_core_model)


if __name__ == "__main__":
    dscnn_checkpoint = "/tf/train_hiFPToi/exp/lstm_ftel5full"
    model_split(
        dscnn_checkpoint + "/model.pb",
        dscnn_checkpoint + "/lstm_ftel5full_fe.tflite",
        dscnn_checkpoint + "/lstm_ftel5full_core.tflite",
        ["decoded_sample_data"],
        ["Mfcc"],
        ["labels_sigmoid"],
    )
