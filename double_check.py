import sys
import subprocess
import shlex

list_fn = "data/trainsets/trainset2_ftel_clone1_devsil/double_check.list"
double_check_fn = "data/trainsets/trainset2_ftel_clone1_devsil/double_check_error.list"

with open(list_fn) as fp:
    lines = fp.readlines()

with open(double_check_fn, "wt") as fp:
    for line in lines:
        toks = line.strip().split()
        if len(toks) == 3:
            print(line)
            ts_e = float(toks[1])
            ts_s = ts_e - 1.3
            while True:
                cmd_play = (
                    "sox "
                    + toks[2]
                    + " -t raw - trim {:.2f} {:.2f} ".format(ts_s, ts_e)
                    + " | aplay -c 1 -t raw -r 16000 -f S16_LE - 2>/dev/null"
                )
                print(cmd_play)
                subprocess.call(cmd_play, shell=True)

                cmd = input(toks[0] + " ?")
                if len(cmd) == 1:
                    if cmd[0] == "n":
                        fp.write(line)
                        break
                    elif cmd[0] == "q":
                        sys.exit(0)
                else:
                    break
