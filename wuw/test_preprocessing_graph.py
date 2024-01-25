import os
import numpy as np
import soundfile as sf
import tensorflow as tf
from utils import load_audio

if __name__ == "__main__":
    preprocessing = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn_ds4a_1/preprocessing.tflite"
    )
    preprocessing.allocate_tensors()
    input_details_preprocessing = preprocessing.get_input_details()
    output_details_preprocessing = preprocessing.get_output_details()

    wav_input = load_audio(
        "2_8_2023/2_8_2023_audio_beam8_phase1_positive/a949d3152ee24869af5ac13d07493c28_0.88_1.92.wav",
        desired_samples=48000,
    )
    wav_input = wav_input.audio
    wav_input = wav_input[11200:32000]

    preprocessing.set_tensor(input_details_preprocessing[0]["index"], wav_input)
    preprocessing.invoke()
    output_data = preprocessing.get_tensor(output_details_preprocessing[0]["index"])
    print(output_data)
