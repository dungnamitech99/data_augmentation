import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.core.framework import summary_pb2
from utils import get_checkpoint_path

import matplotlib.pyplot as plt
import data as wuw_data
import models as wuw_models
import yaml
import json
from tqdm import tqdm
from PIL import Image
import time

tf.compat.v1.disable_eager_execution()
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 100


def train_kws(conf_yaml):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    # print(model.summary())

    conf_yaml["model_conf"] = model.prepare_model_settings()

    fingerprint_size = conf_yaml["model_conf"]["fingerprint_size"]
    label_count = conf_yaml["model_conf"]["label_count"]

    fingerprint_input = tf.compat.v1.placeholder(
        tf.float32, [None, fingerprint_size], name="fingerprint_input"
    )
    # print("--------------------------------")
    # print("fingerprint_input", fingerprint_input)
    # print("--------------------------------")
    ground_truth_input = tf.compat.v1.placeholder(
        tf.float32, [None, label_count], name="groundtruth_input"
    )
    logits, dropout_prob = model.forward(
        fingerprint_input, conf_yaml["model_conf"]["model_size_info"]
    )
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_normed_layer_6_v1")
    )
    # print("----------------------------------")
    # print("logits", logits)
    # print("----------------------------------")

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope("cross_entropy"):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits
            )
        )
    tf.compat.v1.summary.scalar("cross_entropy", cross_entropy_mean)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    # print("-------------------------------------")
    # print("update_ops", update_ops)
    # print("-------------------------------------")
    with tf.name_scope("train"), tf.control_dependencies(update_ops):
        learning_rate_input = tf.compat.v1.placeholder(
            tf.float32, [], name="learning_rate_input"
        )
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate_input).minimize(
            cross_entropy_mean
        )

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count
    )
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar("accuracy", evaluation_step)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    tf.compat.v1.global_variables_initializer().run()
    params = tf.compat.v1.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    print("Total number of Parameters: {}\n-----".format(num_params))
    # Save graph.pbtxt.
    tf.io.write_graph(
        sess.graph_def,
        conf_yaml["exp_conf"]["expdir"],
        "{}.pbtxt".format(conf_yaml["model_conf"]["name"]),
    )

    merged_summaries = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(conf_yaml["exp_conf"]["expdir"], "train"), sess.graph
    )
    validation_writer = tf.compat.v1.summary.FileWriter(os.path.join(expdir, "val"))

    audio_loader = wuw_data.AudioLoader(
        conf_yaml["data_conf"],
        conf_yaml["feat_conf"],
        conf_yaml["model_conf"],
        cache=True,
    )

    FileWriter(conf_yaml["exp_conf"]["logdir"], graph=sess.graph).close()
    training_steps_list = conf_yaml["model_conf"]["training_steps"]
    learning_rates_list = conf_yaml["model_conf"]["learning_rate"]

    augment_pos_to_nev = {"freq": conf_yaml["model_conf"]["augment_pos_to_nev_freq"]}

    augment_mask = {
        "freq_freq": conf_yaml["model_conf"]["augment_mask_freq_freq"],
        "freq_param": conf_yaml["model_conf"]["augment_mask_freq_param"],
        "time_freq": conf_yaml["model_conf"]["augment_mask_time_freq"],
        "time_param": conf_yaml["model_conf"]["augment_mask_time_param"],
    }

    # training loop
    step = 0
    epoch = 0
    best_accuracy = 0
    training_steps_max = np.sum(training_steps_list)
    # print("---------------------------------")
    # print("training_steps_list", training_steps_list)
    # print("---------------------------------")
    train_size = audio_loader.size("training")
    while step < training_steps_max + 1:
        epoch += 1
        audio_loader.shuffle(set_index="training")

        for offset in range(0, train_size, conf_yaml["model_conf"]["batch_size"]):
            ts1 = time.time()
            step += 1
            if step >= training_steps_max + 1:
                break
            training_steps_sum = 0
            learning_rate_value = 0
            for i in range(len(training_steps_list)):
                training_steps_sum += training_steps_list[i]
                if step <= training_steps_sum:
                    learning_rate_value = learning_rates_list[i]
                    break

            # train
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
                mode="training",
            )
            # if step == 1:
            #     # dump features
            #     feats = train_fingerprints[0:10,:].reshape(-1,64,10).transpose(0,2,1).reshape(-1,64)
            #     # print("------------------------------------------")
            #     # print("feats", train_fingerprints[:10,:].shape)
            #     # print("feats", train_fingerprints[:10,:].reshape(-1,64,10).shape)
            #     # print("feats", train_fingerprints[:10,:].reshape(-1,64,10).transpose(0,2,1).shape)
            #     # print("feats", train_fingerprints[0:10,:].reshape(-1,64,10).transpose(0,2,1).reshape(-1,64).shape)
            #     # print("------------------------------------------")
            #     plt.imshow(feats)
            #     images_dir = conf_yaml["exp_conf"]["expdir"]+"/logs/images"
            #     if not os.path.exists(images_dir):
            #         os.makedirs(images_dir)
            #     plt.savefig(images_dir+"/mfcc_"+str(step)+".png")

            train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
                [
                    merged_summaries,
                    evaluation_step,
                    cross_entropy_mean,
                    train_step,
                    increment_global_step,
                ],
                feed_dict={
                    fingerprint_input: train_fingerprints,
                    ground_truth_input: train_ground_truth,
                    learning_rate_input: learning_rate_value,
                    dropout_prob: 1.0,
                },
            )
            train_writer.add_summary(train_summary, step)
            train_writer.flush()
            ts2 = time.time()
            print(
                f"Epoch {epoch} - Step {step} (time {ts2-ts1}): train accuracy {train_accuracy * 100}, "
                f"cross entropy {cross_entropy_value}, lr {learning_rate_value}"
            )
            # val
            # if step % conf_yaml["model_conf"]["eval_step_interval"] == 0:
        if True:
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

                val_summary, val_accuracy, val_matrix, val_cem = sess.run(
                    [
                        merged_summaries,
                        evaluation_step,
                        confusion_matrix,
                        cross_entropy_mean,
                    ],
                    feed_dict={
                        fingerprint_input: val_fingerprints,
                        ground_truth_input: val_ground_truth,
                        dropout_prob: 1.0,
                    },
                )

                # validation_writer.add_summary(val_summary, step)
                batch_size = min(conf_yaml["model_conf"]["batch_size"], val_size - i)
                total_accuracy += (val_accuracy * batch_size) / val_size
                val_cem_mean += (val_cem * batch_size) / val_size

                if total_conf_matrix is None:
                    total_conf_matrix = val_matrix
                else:
                    total_conf_matrix += val_matrix

            summ = summary_pb2.Summary(
                value=[
                    summary_pb2.Summary.Value(
                        tag="accuracy", simple_value=total_accuracy
                    ),
                    summary_pb2.Summary.Value(
                        tag="cross_entropy_1", simple_value=val_cem_mean
                    ),
                ]
            )
            validation_writer.add_summary(summ, step)
            validation_writer.flush()
            print(f"Confusion matrix: \n {total_conf_matrix}")
            print(f"Step {step}: val accuracy {total_accuracy}")

            # Save the model checkpoint when validation accuracy improves
            if total_accuracy >= best_accuracy:
                best_accuracy = total_accuracy
                checkpoint_path = os.path.join(
                    conf_yaml["exp_conf"]["expdir"],
                    "best",
                    "{}_{}.ckpt".format(
                        conf_yaml["model_conf"]["name"], str(int(best_accuracy * 10000))
                    ),
                )
                saver.save(sess, checkpoint_path, global_step=step)
                print(f"Saving best model to {checkpoint_path} - step {step}")
            print(f"So far the best validation accuracy is {best_accuracy}")

    saver.save(
        sess,
        conf_yaml["exp_conf"]["expdir"]
        + "/train/last_step_checkpoint.ckpt-"
        + str(step),
    )


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
