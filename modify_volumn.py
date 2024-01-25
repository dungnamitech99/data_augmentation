import subprocess
import numpy as np
import re
from glob import glob

if __name__ == "__main__":
    negative_paths = glob("negative_3s/negative_3s_quangbd_data/*.wav")

    # with open("data/backgound/all_bg_list_volumn_normed_list.txt", "w") as f_write:
    #     with open("data/backgound/all_bg_list.txt", "r") as f:
    #         for line in f:
    #             path = line.strip()
    #             audio_name = path.split(".")[-2]
    #             destination = audio_name + "_volumn_normed.wav"
    #             random_number = np.random.randint(-30, -9)
    #             print("audio_name: ", audio_name)
    #             subprocess.run(f"sox {path} {destination} norm {random_number}", shell=True, check=True)
    #             f_write.write("/".join(line.split(".")[:-1]) + "_volumn_normed.wav" + "\n")

    # with open("data/trainsets/wuw_ds4a/test15_volumn_normed.list", "w") as f_write:
    #     with open("data/trainsets/wuw_ds4a/test15.list", "r") as f:
    #         for line in f:
    #             path = line.strip().split(" ")[-1]
    #             destination = re.sub(r"\.wav", "_volumn_normed.wav", path)
    #             random_number = np.random.randint(-30, -9)
    #             print("audio_name: ", destination)
    #             subprocess.run(f"sox {path} {destination} norm {random_number}", shell=True, check=True)
    #             f_write.write(" ".join(line.strip().split(" ")[:-1]) + " " + destination + "\n")

    # with open("data/trainsets/wuw_ds4a/val10_volumn_normed.list", "w") as f_write:
    #     with open("data/trainsets/wuw_ds4a/val10.list", "r") as f:
    #         for line in f:
    #             path = line.strip().split(" ")[-1]
    #             destination = re.sub(r"\.wav", "_volumn_normed.wav", path)
    #             random_number = np.random.randint(-30, -9)
    #             print("audio_name: ", destination)
    #             subprocess.run(f"sox {path} {destination} norm {random_number}", shell=True, check=True)
    #             f_write.write(" ".join(line.strip().split(" ")[:-1]) + " " + destination + "\n")

    # with open("data/trainsets/wuw_ds4a/train75_volumn_normed.list", "w") as f_write:
    #     with open("data/trainsets/wuw_ds4a/train75.list", "r") as f:
    #         for line in f:
    #             path = line.strip().split(" ")[-1]
    #             destination = re.sub(r"\.wav", "_volumn_normed.wav", path)
    #             random_number = np.random.randint(-30, -9)
    #             print("audio_name: ", destination)
    #             subprocess.run(f"sox {path} {destination} norm {random_number}", shell=True, check=True)
    #             f_write.write(" ".join(line.strip().split(" ")[:-1]) + " " + destination + "\n")

    with open("data/trainsets/wuw_ds4a/negative_volumn_normed.list", "w") as f_write:
        for path in negative_paths:
            path = path.strip()
            destination = re.sub(r"\.wav", "_volumn_normed.wav", path)
            random_number = np.random.randint(-30, -9)
            print("audio_name: ", destination)
            subprocess.run(
                f"sox {path} {destination} norm {random_number}", shell=True, check=True
            )
            f_write.write(destination + "\n")
