import sys
import os
import random

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(sys.argv[0], " <in_data_dir> <dev_per> <out_train_dir> <out_dev_dir>")
        print("eg. ", sys.argv[0], " data/train 20 data/train80 data/dev20")
        sys.exit()

    in_data_dir = sys.argv[1]
    dev_per = int(sys.argv[2])
    out_train_dir = sys.argv[3]
    out_dev_dir = sys.argv[4]

    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir, exist_ok=True)
    if not os.path.exists(out_dev_dir):
        os.makedirs(out_dev_dir, exist_ok=True)

    uttids = []
    with open(in_data_dir + "/uttid", "rt") as fp:
        for line in fp:
            line = line.strip()
            if len(line) > 0:
                uttids.append(line)

    random.shuffle(uttids)
    N = dev_per * len(uttids) // 100

    dev_ids = uttids[0:N]
    dev_ids.sort()
    train_ids = uttids[N:]
    train_ids.sort()

    with open(out_train_dir + "/uttid", "wt") as fp:
        fp.write("\n".join(train_ids))

    with open(out_dev_dir + "/uttid", "wt") as fp:
        fp.write("\n".join(dev_ids))

    wavfns = {}
    with open(in_data_dir + "/wav.scp", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) == 2:
                wavfns[toks[0]] = toks[1]

    with open(out_train_dir + "/wav.scp", "wt") as fp:
        for uttid in train_ids:
            if uttid in wavfns.keys():
                fp.write(uttid + " " + wavfns[uttid] + "\n")
            else:
                print("Missing ", uttid)

    with open(out_dev_dir + "/wav.scp", "wt") as fp:
        for uttid in dev_ids:
            if uttid in wavfns.keys():
                fp.write(uttid + " " + wavfns[uttid] + "\n")
            else:
                print("Missing ", uttid)

    calib = {}
    with open(in_data_dir + "/calib.txt", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) >= 2:
                calib[toks[0]] = toks[1:]

    with open(out_train_dir + "/calib.txt", "wt") as fp:
        for uttid in train_ids:
            if uttid in calib.keys():
                s = " ".join(calib[uttid])
                fp.write(uttid + " " + s + "\n")
            else:
                print("Missing ", uttid)

    with open(out_dev_dir + "/calib.txt", "wt") as fp:
        for uttid in dev_ids:
            if uttid in calib.keys():
                s = " ".join(calib[uttid])
                fp.write(uttid + " " + s + "\n")
            else:
                print("Missing ", uttid)

    labels = {}
    with open(in_data_dir + "/labels", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) == 2:
                labels[toks[0]] = toks[1]

    with open(out_train_dir + "/labels", "wt") as fp:
        for uttid in train_ids:
            if uttid in labels.keys():
                fp.write(uttid + " " + labels[uttid] + "\n")
            else:
                print("Missing ", uttid)

    with open(out_dev_dir + "/labels", "wt") as fp:
        for uttid in dev_ids:
            if uttid in labels.keys():
                fp.write(uttid + " " + labels[uttid] + "\n")
            else:
                print("Missing ", uttid)

    with open(out_train_dir + "/label_time_wav.txt", "wt") as fp:
        for uttid in train_ids:
            if (
                uttid in labels.keys()
                and uttid in wavfns.keys()
                and uttid in calib.keys()
            ):
                fp.write(
                    labels[uttid] + " " + calib[uttid][0] + " " + wavfns[uttid] + "\n"
                )
            else:
                print("Missing ", uttid)

    with open(out_dev_dir + "/label_time_wav.txt", "wt") as fp:
        for uttid in dev_ids:
            if (
                uttid in labels.keys()
                and uttid in wavfns.keys()
                and uttid in calib.keys()
            ):
                fp.write(
                    labels[uttid] + " " + calib[uttid][0] + " " + wavfns[uttid] + "\n"
                )
            else:
                print("Missing ", uttid)
