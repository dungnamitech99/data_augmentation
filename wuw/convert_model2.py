import yaml
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
import soundfile as sf
import models as wuw_models
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.python.framework import graph_util
from utils import get_checkpoint_path, load_audio
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_audio

if __name__ == "__main__":
    tf.disable_eager_execution()
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    with open("/tf/train_hiFPToi/conf/dscnn-l.yaml", "r") as conf_fp:
        conf_yaml = yaml.safe_load(conf_fp)
    model = wuw_models.select_model(
        conf_yaml["data_conf"], conf_yaml["feat_conf"], conf_yaml["model_conf"]
    )
    model_settings = model.prepare_model_settings()
    fingerprint_size = conf_yaml["model_conf"]["fingerprint_size"]
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name="fingerprint_input"
    )
    logits, dropout_prob = model.forward(
        fingerprint_input, conf_yaml["model_conf"]["model_size_info"], is_training=False
    )
    model.load_variables_from_checkpoint(
        sess, get_checkpoint_path("/tf/train_hiFPToi/exp/dscnn_normed_layer_6_v1")
    )
    tf.sigmoid(logits, name="labels_sigmoid")
    nodes = [op.name for op in tf.get_default_graph().get_operations()]
    print(nodes)

    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["labels_sigmoid"]
    )
    pb_path = os.path.join("/tf/train_hiFPToi/exp/dscnn_normed_layer_6_v1", "model.pb")
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname(pb_path),
        os.path.basename(pb_path),
        as_text=False,
    )
    print(f"Saved frozen graph to {pb_path}")

    # to tflite
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        pb_path, input_arrays=["fingerprint_input"], output_arrays=["labels_sigmoid"]
    )
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    with open(
        os.path.join("/tf/train_hiFPToi/exp/dscnn_normed_layer_6_v1", "model.tflite"),
        "wb",
    ) as f:
        f.write(tflite_model)
