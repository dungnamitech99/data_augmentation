import sys
import time
import yaml
import data as wuw_data
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import tensorflow_io as tfio
from tqdm import tqdm
import numpy as np
import models as wuw_models
import matplotlib.pyplot as plt
from utils import get_checkpoint_path
from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary.writer.writer import FileWriter

tf.disable_eager_execution()
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.dpi"] = 100
sess = tf.InteractiveSession()


def evaluate_models(conf_yaml):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )

    conf_yaml["model_conf"] = model.prepare_model_settings()

    fingerprint_size = conf_yaml["model_conf"]["fingerprint_size"]
    label_count = conf_yaml["model_conf"]["label_count"]

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name="fingerprint_input"
    )
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name="groundtruth_input"
    )
    logits, dropout_prob = model.forward(
        fingerprint_input, conf_yaml["model_conf"]["model_size_info"], is_training=False
    )
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_ds4a_1")
    )

    with tf.name_scope("cross_entropy"):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits
            )
        )
    tf.summary.scalar("cross_entropy", cross_entropy_mean)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.math.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count
    )
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", evaluation_step)

    # tf.global_variables_initializer().run()
    # params = tf.trainable_variables()
    # num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    # print('Total number of Parameters: {}\n-----'.format(num_params))

    merged_summaries = tf.summary.merge_all()

    audio_loader = wuw_data.AudioLoader(
        conf_yaml["data_conf"],
        conf_yaml["feat_conf"],
        conf_yaml["model_conf"],
        cache=True,
    )

    val_cem_mean = 0
    total_accuracy = 0
    val_size = audio_loader.size("validation")
    total_conf_matrix = None
    print(
        "-------------------hihi------------------",
    )
    for i in tqdm(range(0, val_size, conf_yaml["model_conf"]["batch_size"])):
        print("-------------------------------------", i)
        val_fingerprints, val_ground_truth, mel = audio_loader.load_batch(
            sess,
            conf_yaml["model_conf"]["batch_size"],
            offset=i,
            background_frequency=0,
            background_volume_range=0,
            time_shift=0,
            mode="validation",
        )

        print("val_fingerprints: ", val_fingerprints.shape)
        print(val_fingerprints)
        print("val_ground_truth: ", val_ground_truth.shape)
        print(val_ground_truth)

        val_summary, val_accuracy, val_matrix, val_cem = sess.run(
            [merged_summaries, evaluation_step, confusion_matrix, cross_entropy_mean],
            feed_dict={
                fingerprint_input: val_fingerprints,
                ground_truth_input: val_ground_truth,
            },
        )

        # validation_writer.add_summary(val_summary, step)
        batch_size = min(conf_yaml["model_conf"]["batch_size"], val_size - i)
        total_accuracy += (val_accuracy * batch_size) / val_size
        val_cem_mean += (val_cem * batch_size) / val_size

        print("total_conf_matrix: ", val_matrix)
        if total_conf_matrix is None:
            total_conf_matrix = val_matrix
        else:
            total_conf_matrix += val_matrix

        # print(mel)

    print(f"Confusion matrix: \n {total_conf_matrix}")
    print(f"Val accuracy: {total_accuracy}")

    # test_cem_mean = 0
    # total_accuracy = 0
    # test_size = audio_loader.size('testing')
    # total_conf_matrix = None
    # print("-------------------hihi------------------",)
    # for i in tqdm(range(0, test_size, conf_yaml["model_conf"]["batch_size"])):
    #     print("-------------------------------------", i)
    #     test_fingerprints, test_ground_truth = audio_loader \
    #         .load_batch(sess, conf_yaml["model_conf"]["batch_size"], offset=i, background_frequency=0,
    #                     background_volume_range=0, time_shift=0, mode='testing')

    #     test_summary, test_accuracy, test_matrix, test_cem = sess.run(
    #         [merged_summaries, evaluation_step, confusion_matrix, cross_entropy_mean],
    #         feed_dict={
    #             fingerprint_input: test_fingerprints,
    #             ground_truth_input: test_ground_truth})

    #     #validation_writer.add_summary(val_summary, step)
    #     batch_size = min(conf_yaml["model_conf"]["batch_size"], test_size - i)
    #     total_accuracy += (test_accuracy * batch_size) / test_size
    #     test_cem_mean += (test_cem * batch_size) / test_size

    #     print("total_conf_matrix: ", test_matrix)
    #     if total_conf_matrix is None:
    #         total_conf_matrix = test_matrix
    #     else:
    #         total_conf_matrix += test_matrix

    # print(f'Confusion matrix: \n {total_conf_matrix}')
    # print(f'Test accuracy: {total_accuracy}')


if __name__ == "__main__":
    with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", "r") as conf_fp:
        conf_yaml = yaml.safe_load(conf_fp)

    evaluate_models(conf_yaml)
