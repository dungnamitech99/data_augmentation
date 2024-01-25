import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import tensorflow.compat.v1 as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.metrics import confusion_matrix
from tensorflow.python.platform import gfile
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.core.framework import summary_pb2

import matplotlib.pyplot as plt
import data as wuw_data
import models as wuw_models
import yaml
import json
from tqdm import tqdm
from PIL import Image
import time
from absl import logging
from utils import save_model_summary
import pprint

tf.compat.v1.disable_eager_execution()
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 100


def train_kws(conf_yaml):
    training_conf = conf_yaml["training_conf"]
    model_conf = conf_yaml["model_conf"]

    logging.set_verbosity(training_conf["verbosity"])
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    config = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    if conf_yaml["model_conf"]["name"] == "bc_resnet":
        model = config.build()
    else:
        model = config.build(conf_yaml["model_conf"]["model_size_info"])
    logging.info(model.summary())

    if conf_yaml["model_conf"]["name"] in ["crnn", "gru", "lstm"]:
        tf.compat.v1.experimental.output_all_intermediates(True)

    # save_model_summary(model, "/tf/train_hiFPToi")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=not model_conf["return_softmax"]
    )

    optimizer = tf.keras.optimizers.Adam(epsilon=training_conf["optimizer_epsilon"])

    if training_conf["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(epsilon=training_conf["optimizer_epsilon"])
    elif training_conf["optimizer"] == "momentum":
        optimizer = tf.keras.optimizers.SGD(mometum=training_conf["momentum"])
    elif training_conf["optimizer"] == "novograd":
        optimizer = tf.keras.optimizers.NovoGrad(
            lr=0.05,
            beta_1=training_conf["novograd_beta_1"],
            beta_2=training_conf["novograd_beta_2"],
            weight_decay=training_conf["novograd_weight_decay"],
            grad_averaging=bool(training_conf["novograd_grad_averaging"]),
        )
    else:
        raise ValueError("Unsupported optimizer: %s" % training_conf["optimizer"])

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    if not os.path.exists(os.path.join(conf_yaml["exp_conf"]["expdir"], "train")):
        os.makedirs(os.path.join(conf_yaml["exp_conf"]["expdir"], "train"))

    if not os.path.exists(os.path.join(conf_yaml["exp_conf"]["expdir"], "validation")):
        os.makedirs(os.path.join(conf_yaml["exp_conf"]["expdir"], "validation"))

    train_writer = tf.summary.FileWriter(
        os.path.join(conf_yaml["exp_conf"]["expdir"], "train"), sess.graph
    )
    validation_writer = tf.summary.FileWriter(
        os.path.join(conf_yaml["exp_conf"]["expdir"], "validation")
    )

    start_step = 1
    logging.info("Training from step: %d", start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, conf_yaml["exp_conf"]["expdir"], "graph.pbtxt")

    best_accuracy = 0.0

    # Prepare parameters for exp learning rate decay
    training_steps_list = model_conf["training_steps"]
    learning_rates_list = model_conf["learning_rate"]
    training_steps_max = np.sum(training_steps_list)
    lr_init = learning_rates_list[0]
    exp_rate = -np.log(learning_rates_list[-1] / lr_init) / training_steps_max

    # Configure checkpointer
    checkpoint_directory = os.path.join(conf_yaml["exp_conf"]["expdir"], "restore")
    checkpoint_prefix = os.path.join(conf_yaml["exp_conf"]["expdir"], "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    sess.run(tf.global_variables_initializer())
    status.initialize_or_restore(sess)

    audio_loader = wuw_data.AudioLoader(
        conf_yaml["data_conf"],
        conf_yaml["feat_conf"],
        conf_yaml["model_conf"],
        cache=True,
    )

    augment_pos_to_nev = {"freq": conf_yaml["model_conf"]["augment_pos_to_nev_freq"]}
    # augment_pos_to_nev: 0.2

    augment_mask = {
        "freq_freq": conf_yaml["model_conf"]["augment_mask_freq_freq"],
        "freq_param": conf_yaml["model_conf"]["augment_mask_freq_param"],
        "time_freq": conf_yaml["model_conf"]["augment_mask_time_freq"],
        "time_param": conf_yaml["model_conf"]["augment_mask_time_param"],
    }

    # Training loop.
    step = 0
    epoch = 0
    best_accuracy = 0
    train_size = audio_loader.size("training")
    print("-------------------------------------")
    print("batch_size: ", conf_yaml["model_conf"]["batch_size"])
    print("-------------------------------------")
    while step < training_steps_max + 1:
        epoch += 1
        audio_loader.shuffle(set_index="training")
        print("111111111111111111111111111111111111111111111111111111111111111")
        for offset in range(0, train_size, conf_yaml["model_conf"]["batch_size"]):
            ts1 = time.time()
            step += 1
            if step >= training_steps_max + 1:
                break

            if training_conf["lr_schedule"] == "exp":
                learning_rate_value = lr_init * np.exp(-exp_rate * (step + 1))
            elif training_conf["lr_schedule"] == "linear":
                training_steps_sum = 0
                for i in range(len(training_steps_list)):
                    training_steps_sum += training_steps_list[i]
                    if step <= training_steps_sum:
                        learning_rate_value = learning_rates_list[i]
                        break
            else:
                raise ValueError("lr_schedule does not exist")

            tf.keras.backend.set_value(model.optimizer.lr, learning_rate_value)
            train_fingerprints, train_ground_truth = audio_loader.load_batch(
                sess,
                batch_size=conf_yaml["model_conf"]["batch_size"],
                offset=offset,
                background_frequency=conf_yaml["data_conf"]["background_frequency"],
                background_volume_range=conf_yaml["data_conf"]["background_volume"],
                background_silence_frequency=conf_yaml["data_conf"][
                    "background_silence_frequency"
                ],
                background_silence_volume_range=conf_yaml["data_conf"][
                    "background_silence_volume"
                ],
                time_shift=conf_yaml["data_conf"]["time_shift"],
                down_volume_frequency=0.1,
                down_volume_range=0.9,
                augment_pos_to_nev=augment_pos_to_nev,
                augment_mask=augment_mask,
            )

            inputs = tf.reshape(
                train_fingerprints,
                [
                    -1,
                    model_conf["spectrogram_length"],
                    model_conf["dct_coefficient_count"],
                ],
            )
            labels = tf.argmax(train_ground_truth, 1)
            # print("train_ground_truth: ", labels)
            # print("train_ground_truth: ", labels.shape)
            result = model.train_on_batch(inputs, labels)
            summary = tf.Summary(
                value=[
                    tf.Summary.Value(tag="accuracy", simple_value=result[1]),
                    tf.Summary.Value(tag="cross_entropy", simple_value=result[0]),
                ]
            )

            train_writer.add_summary(summary, step)
            ts2 = time.time()
            print(
                f"Epoch {epoch} - Step {step} (time {ts2-ts1}): train accuracy {result[1] * 100}, "
                f"cross entropy {result[0]}, lr {learning_rate_value}"
            )
        print("111111111111111111111111111111111111111111111111111111111111111")
        if True:
            print(
                "---------------------------------------------------------->Validation"
            )
            val_cem_mean = 0
            total_accuracy = 0
            val_size = audio_loader.size("validation")
            total_conf_matrix = None
            for i in tqdm(range(0, val_size, conf_yaml["model_conf"]["batch_size"])):
                val_fingerprints, val_ground_truth = audio_loader.load_batch(
                    sess,
                    conf_yaml["model_conf"]["batch_size"],
                    offset=i,
                    background_frequency=0,
                    background_volume_range=0,
                    time_shift=0,
                    mode="validation",
                )

                val_inputs = tf.reshape(
                    val_fingerprints,
                    [
                        -1,
                        model_conf["spectrogram_length"],
                        model_conf["dct_coefficient_count"],
                    ],
                )
                val_labels = tf.argmax(val_ground_truth, 1)
                result = model.test_on_batch(val_inputs, val_labels)

                batch_size = min(conf_yaml["model_conf"]["batch_size"], val_size - i)
                total_accuracy += (result[1] * batch_size) / val_size
                val_cem_mean += (result[0] * batch_size) / val_size

                logits = model.predict_on_batch(val_inputs)
                predicts = tf.argmax(logits, 1)
                # confusion_matrix = tf.math.confusion_matrix(val_labels, predicts, num_classes=model_conf["label_count"])
                # if total_conf_matrix is None:
                #     total_conf_matrix = confusion_matrix
                # else:
                #     total_conf_matrix += confusion_matrix

            summary = tf.Summary(
                value=[
                    tf.Summary.Value(tag="accuracy", simple_value=total_accuracy),
                    tf.Summary.Value(tag="cross_entropy_1", simple_value=val_cem_mean),
                ]
            )

            validation_writer.add_summary(summary, step)
            print(f"Confusion matrix: \n {total_conf_matrix}")
            print(f"Step {step}: val accuracy {total_accuracy}")

            if total_accuracy >= best_accuracy:
                best_accuracy = total_accuracy
                if not os.path.exists(
                    os.path.join(conf_yaml["exp_conf"]["expdir"], "best")
                ):
                    os.makedirs(os.path.join(conf_yaml["exp_conf"]["expdir"], "best"))
                checkpoint_path = os.path.join(
                    conf_yaml["exp_conf"]["expdir"],
                    "best",
                    "{}_{}.ckpt".format(
                        conf_yaml["model_conf"]["name"], str(int(best_accuracy * 10000))
                    ),
                )
                # model.save(checkpoint_path)
                # model.save_weights(os.path.join(conf_yaml["exp_conf"]["expdir"], "best", "{}_{}.ckpt".format(conf_yaml["model_conf"]["name"], str(int(best_accuracy*1000)))))
                model.save_weights(
                    os.path.join(conf_yaml["exp_conf"]["expdir"], "best_weights")
                )

                # Save checkpoint
                checkpoint.save(file_prefix=checkpoint_prefix, session=sess)
                print(f"Saving best model to {checkpoint_path} - step {step}")
            print(f"So far the best validation accuracy is {best_accuracy}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(sys.argv[0], " <conf> <expdir>")
        sys.exit(1)

    # conf_fn = "conf/lstm-s.yaml"
    # expdir = "exp/utt3s_trainset2_lstm_step1k_allbg5_bsize2k"
    # conf_fn = "conf/dscnn-tiny.yaml"
    # expdir = "exp/utt3s_trainset2_dscnn-tiny_step10k_allbg5"

    conf_fn = sys.argv[1]
    expdir = sys.argv[2]
    logdir = expdir + "/logs"

    with open(conf_fn, "r") as conf_fp:
        try:
            conf_yaml = yaml.safe_load(conf_fp)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(2)
    # print("Before:" )
    # print("---------------------")
    # print(conf_yaml["model_conf"]["model_size_info"])
    # print("---------------------")

    conf_yaml["exp_conf"] = {"logdir": logdir, "expdir": expdir}

    # print("After:" )
    # print("---------------------")
    # print(conf_yaml)
    # print("---------------------")

    train_kws(conf_yaml)
