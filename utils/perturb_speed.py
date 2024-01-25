import argparse
import os
import subprocess


def perturb_speed_fn(factor, in_fn, out_fn):
    if factor <= 1:
        cmd = (
            "sox " + in_fn + " " + out_fn + " speed {:.3f}".format(factor) + " trim 0 3"
        )
    else:
        cmd = (
            "sox "
            + in_fn
            + " "
            + out_fn
            + " speed {:.3f}".format(factor)
            + " pad 0 {:.3f}".format(3.0 - 3.0 / factor)
        )
    # print(cmd)
    subprocess.call(cmd, shell=True)


def perturb_speed_list(args):
    if not os.path.isdir(args.out_wav_dir):
        os.makedirs(args.out_wav_dir)

    with open(args.input_list_fn) as fp:
        lines = fp.readlines()

    with open(args.output_list_fn, "wt") as out_fp:
        for line in lines:
            toks = line.strip().split()
            if len(toks) == 3:
                uttid = toks[0]
                ts_e = float(toks[1]) / args.factor
                wav_fn = toks[2]
                base_fn = os.path.basename(wav_fn)
                out_wav_fn = args.out_wav_dir + "/" + base_fn
                perturb_speed_fn(args.factor, wav_fn, out_wav_fn)

                out_line = uttid + " {:.2f}".format(ts_e) + " " + out_wav_fn + "\n"
                out_fp.write(out_line)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="perturb speed of speech data")

    # Add arguments
    parser.add_argument(
        "-f",
        "--factor",
        type=float,
        required=True,
        help="A floating factor for adjusting the playback speed",
    )
    parser.add_argument(
        "-i",
        "--input-list-fn",
        type=str,
        required=True,
        help="input label_time_wav.txt",
    )
    parser.add_argument(
        "-d", "--out-wav-dir", type=str, required=True, help="output wav dir"
    )
    parser.add_argument(
        "-o",
        "--output-list-fn",
        type=str,
        required=True,
        help="output label_time_wav.txt",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    perturb_speed_list(args)
