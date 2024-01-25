import sys
import os
import random

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(sys.argv[0], " <in_data_dir>")
        print("eg. ", sys.argv[0], " data/train")
        sys.exit()

    in_data_dir = sys.argv[1]

    uttids = []
    with open(in_data_dir + "/uttid", "rt") as fp:
        for line in fp:
            line = line.strip()
            if len(line) > 0:
                uttids.append(line)

    wavfns = {}
    with open(in_data_dir + "/wav.scp", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) == 2:
                wavfns[toks[0]] = toks[1]

    labels = {}
    with open(in_data_dir + "/labels", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) == 2:
                labels[toks[0]] = toks[1]

    calib = {}
    with open(in_data_dir + "/calib.txt", "rt") as fp:
        for line in fp:
            toks = line.strip().split()
            if len(toks) >= 2:
                calib[toks[0]] = toks[1]

    with open(in_data_dir + "/label_time_wav.txt", "wt") as fp:
        for uttid in uttids:
            if (
                uttid in labels.keys()
                and uttid in wavfns.keys()
                and uttid in calib.keys()
            ):
                fp.write(
                    labels[uttid] + " " + calib[uttid] + " " + wavfns[uttid] + "\n"
                )
            else:
                print("Missing ", uttid)
