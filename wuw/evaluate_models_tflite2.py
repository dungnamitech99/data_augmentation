import os
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
from utils import load_audio
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix


if __name__ == "__main__":
    # initialize preprocessing
    preprocessing = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn/dscnn-add-ftel5-mfcc10-mvn_fe.tflite"
    )
    preprocessing.allocate_tensors()
    input_details_preprocessing = preprocessing.get_input_details()
    output_details_preprocessing = preprocessing.get_output_details()

    # initialize model
    model = tf.lite.Interpreter(
        model_path="/tf/train_hiFPToi/exp/dscnn-add-ftel5-mfcc10-mvn/dscnn-add-ftel5-mfcc10-mvn_core.tflite"
    )
    model.allocate_tensors()
    input_details_model = model.get_input_details()
    output_details_model = model.get_output_details()

    # get audio path
    dic = {"positive": [], "negative": []}

    with open("/tf/train_hiFPToi/ftel5_5Dec2023/data_bugs.txt", "r") as f:
        data_bugs = f.readlines()

    data_bugs = [data_path.strip() for data_path in data_bugs]
    with open("/tf/train_hiFPToi/negative_record/negative_record_test.txt", "r") as f:
    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val_new_add_ftel5.txt", "r") as f:
        for line in f:
            components = line.strip().split()
            path_components = components[-1].split("/")
            # new_path = path_components[0] + "_add_noise/" +  "/".join(path_components[1:])
            new_path = path_components[0] + "/" + "/".join(path_components[1:])
            # if path in data_bugs:
            #     continue
            if components[0] == "positive":
                dic["positive"].append(new_path)
            else:
                dic["negative"].append(new_path)
    # with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train_part_9.txt", "r") as f:
    #     for line in f:
    #         components = line.strip().split()
    #         if components[0] == "positive":
    #             dic["positive"].append(components[-1])
    #         else:
    #             dic["negative"].append(components[-1])

    # with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train_part_10.txt", "r") as f:
    #     for line in f:
    #         components = line.strip().split()
    #         if components[0] == "positive":
    #             dic["positive"].append(components[-1])
    #         else:
    #             dic["negative"].append(components[-1])

    # with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train_part_11.txt", "r") as f:
    #     for line in f:
    #         components = line.strip().split()
    #         if components[0] == "positive":
    #             dic["positive"].append(components[-1])
    #         else:
    #             dic["negative"].append(components[-1])

    # evaluate
    labels = []
    predicts = []
    mark_sample = 2 * 16000
    shift_offset = int(0.01 * 16000)
    input_len = int(1.3 * 16000)
    result = {}
    print("----------------------------------------------------------------------")
    for label in dic:
        print("label: ", label)
        for path in tqdm(dic[label], total=len(dic[label])):
            # if "audio_beam8_phase1_fa1/" not in path:
            print("path: ", path)
            labels.append(label)
            wav_input = load_audio(path, desired_samples=48000)
            print(
                "---------------------------------hihi---------------------------------------"
            )
            wav_input = wav_input.audio
            wav_input = wav_input[11200:32000]

            # feed data to preprocessing graph
            preprocessing.set_tensor(input_details_preprocessing[0]["index"], wav_input)
            preprocessing.invoke()
            output_data = preprocessing.get_tensor(
                output_details_preprocessing[0]["index"]
            )

            print(output_data.shape)
            # feed data to wuw models
            model.set_tensor(input_details_model[0]["index"], output_data)
            model.invoke()
            output_data = model.get_tensor(output_details_model[0]["index"])
            predicts.append(["positive", "negative"][np.argmax(output_data)])
            result[path] = list(output_data[0].astype(float))

    with open(
        "/tf/train_hiFPToi/analysis_results/result_tflite_dscnn_negative_recorded_24012024.json",
        "w",
    ) as f:
        json.dump(result, f)

    print(len(labels))
    print(len(predicts))
    print(classification_report(labels, predicts))
    print(confusion_matrix(labels, predicts))
