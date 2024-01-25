import os
import numpy as np
import soundfile as sf
import tensorflow as tf


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
    preprocessing = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/test_bcresnet_2/preprocessing.tflite"
    )
    preprocessing.allocate_tensors()
    input_details_preprocessing = preprocessing.get_input_details()
    output_details_preprocessing = preprocessing.get_output_details()
    # print(input_details)
    # print(output_details)
    # input_shape = input_details[0]["shape"]
    # print(input_shape)

    x, sr = audioread(
        "/tf/train_hiFPToi/waves/train/ftel_train/positive_3s_isolate_clone_musicV30_16k_processed/wav/0a1c1122bb204c10a10348d41dc5beeb_1.96_3.84.wav"
    )
    input_data = np.expand_dims(x, 1)
    input_data = input_data[11200:32000]
    input_data = input_data.astype(np.float32)
    preprocessing.set_tensor(input_details_preprocessing[0]["index"], input_data)
    preprocessing.invoke()
    output_data = preprocessing.get_tensor(output_details_preprocessing[0]["index"])
    # print(output_data.shape)

    interpreter = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/test_bcresnet_2/bc_resnet.tflite"
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], output_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(["positive", "negative"][np.argmax(output_data)])
    print(output_data)
