import os
import yaml
import random
import argparse
import numpy as np
from glob import glob
import tensorflow as tf
from types import SimpleNamespace


def set_seed(seed=1995):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def prepare_config():
    parser = argparse.ArgumentParser(description="set input arguments")
    parser.add_argument("--config", type=str, default="configs/lstm-s.yaml")
    parser.add_argument("--command", type=str, default="train")
    args = parser.parse_args()
    with open(args.config) as input_file:
        config = yaml.load(input_file, Loader=yaml.FullLoader)["config"]
        if args.command == "train":
            config["train"] = True
        elif args.command == "convert":
            config["convert"] = True
        elif args.command == "record":
            config["record"] = True
        else:
            raise Exception("Command is invalid [train, convert, record]")
        return SimpleNamespace(**config)


def get_exp_dir(args):
    return f"exp/{args.model_architecture}_{'_'.join([str(i) for i in args.model_size_info])}"


def get_checkpoint_path(train_dir):
    checkpoint_path = train_dir + "/best/checkpoint"
    with open(checkpoint_path, "rt") as fp:
        lines = fp.readlines()

    # print(lines)
    toks = lines[-1].strip().split()
    print("toks: ", toks)

    if len(toks) == 2:
        return train_dir + "/best/" + toks[1][1:-1]
    return ""


def get_checkpoint_path_Quang(train_dir):
    checkpoint_dir = os.path.join(train_dir, "best", "*ckpt*.index")
    max_epoch = 0
    result = None
    for fpath in glob(checkpoint_dir):
        epoch = int(fpath.split(".")[1].split("-")[1])
        if max_epoch < epoch:
            result = fpath
            max_epoch = epoch
    return ".".join(result.split(".")[:-1])


def load_audio(path, desired_samples):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        wav_filename_placeholder = tf.compat.v1.placeholder(tf.string, [])
        wav_loader = tf.io.read_file(wav_filename_placeholder)
        wav_decoder = tf.audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples
        )
        wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: path})
    return wav_data
