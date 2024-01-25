import re
import argparse
import random
from glob import glob


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--balanced", type=int)
    # args = parser.parse_args()

    # positive_paths = glob("waves/ftel4/positive/2_8_2023_audio_beam8_phase1_positive/*.wav") \
    #                 + glob("waves/ftel4/positive/3_8_2023_audio_beam8_phase1_positive/*.wav")

    # negative_paths = glob("waves/ftel4/negative/2_8_2023_fa/**/*.wav", recursive=True) \
    #                 + glob("waves/ftel4/negative/3_8_2023_fa/**/*.wav", recursive=True) \

    # silence_paths = glob("waves/ftel4/silence/wav/*.wav")

    # print(len(positive_paths))
    # print(len(negative_paths))
    # print(len(silence_paths))

    # data_balanced_train = []
    # data_balanced_val = []
    # data_balanced_test = []
    # data_unbalanced_train = []
    # data_unbalanced_val = []
    # data_unbalanced_test = []

    # if args.balanced:
    #     sub_negative_paths = negative_paths[:65000]

    #     # Train set
    #     for path in positive_paths[int(0.3*len(positive_paths)):len(positive_paths)]:
    #         data_balanced_train.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in sub_negative_paths[int(0.3*len(sub_negative_paths)):len(sub_negative_paths)]:
    #         data_balanced_train.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[int(0.3*len(silence_paths)):len(silence_paths)]:
    #         data_balanced_train.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     # Val set
    #     for path in positive_paths[int(0.15*len(positive_paths)):int(0.3*len(positive_paths))]:
    #         data_balanced_val.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in sub_negative_paths[int(0.15*len(sub_negative_paths)):int(0.3*len(sub_negative_paths))]:
    #         data_balanced_val.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[int(0.15*len(silence_paths)):int(0.3*len(silence_paths))]:
    #         data_balanced_val.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     # Test set
    #     for path in positive_paths[:int(0.15*len(positive_paths))]:
    #         data_balanced_test.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in sub_negative_paths[:int(0.15*len(sub_negative_paths))]:
    #         data_balanced_test.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[:int(0.15*len(silence_paths))]:
    #         data_balanced_test.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     # Shuffle data
    #     random.shuffle(data_balanced_train)
    #     random.shuffle(data_balanced_val)
    #     random.shuffle(data_balanced_test)

    #     with open("data/trainsets/trainset4_ftel4_balanced/label_time_wav_70.txt", "w") as f:
    #         for line in data_balanced_train:
    #             f.write(line)

    #     with open("data/trainsets/trainset4_ftel4_balanced/label_time_wav_val_15.txt", "w") as f:
    #         for line in data_balanced_val:
    #             f.write(line)

    #     with open("data/trainsets/trainset4_ftel4_balanced/label_time_wav_test_15.txt", "w") as f:
    #         for line in data_balanced_test:
    #             f.write(line)

    #     print(len(positive_paths[int(0.3*len(positive_paths)):len(positive_paths)]))
    #     print(len(positive_paths[int(0.15*len(positive_paths)):int(0.3*len(positive_paths))]))
    #     print(len(positive_paths[:int(0.15*len(positive_paths))]))
    #     print("------------------------------------------------------------------------------")
    #     print(len(sub_negative_paths[int(0.3*len(sub_negative_paths)):len(sub_negative_paths)]))
    #     print(len(sub_negative_paths[int(0.15*len(sub_negative_paths)):int(0.3*len(sub_negative_paths))]))
    #     print(len(sub_negative_paths[:int(0.15*len(sub_negative_paths))]))
    #     print("------------------------------------------------------------------------------")
    #     print(len(silence_paths[int(0.3*len(silence_paths)):len(silence_paths)]))
    #     print(len(silence_paths[int(0.15*len(silence_paths)):int(0.3*len(silence_paths))]))
    #     print(len(silence_paths[:int(0.15*len(silence_paths))]))

    # else:
    #     # print(len(positive_paths[int(0.3*len(positive_paths)):len(positive_paths)]))
    #     # print(len(positive_paths[int(0.15*len(positive_paths)):int(0.3*len(positive_paths))]))
    #     # print(len(positive_paths[:int(0.15*len(positive_paths))]))
    #     # print("------------------------------------------------------------------------------")
    #     # print(len(negative_paths[int(0.3*len(negative_paths)):len(negative_paths)]))
    #     # print(len(negative_paths[int(0.15*len(negative_paths)):int(0.3*len(negative_paths))]))
    #     # print(len(negative_paths[:int(0.15*len(negative_paths))]))
    #     # print("------------------------------------------------------------------------------")
    #     # print(len(silence_paths[int(0.3*len(silence_paths)):len(silence_paths)]))
    #     # print(len(silence_paths[int(0.15*len(silence_paths)):int(0.3*len(silence_paths))]))
    #     # print(len(silence_paths[:int(0.15*len(silence_paths))]))
    #     # Train set

    #     for path in positive_paths[int(0.3*len(positive_paths)):len(positive_paths)]:
    #         data_unbalanced_train.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in negative_paths[int(0.3*len(negative_paths)):len(negative_paths)]:
    #         data_unbalanced_train.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[int(0.3*len(silence_paths)):len(silence_paths)]:
    #         data_unbalanced_train.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     # Val set

    #     for path in positive_paths[int(0.15*len(positive_paths)):int(0.3*len(positive_paths))]:
    #         data_unbalanced_val.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in negative_paths[int(0.15*len(negative_paths)):int(0.3*len(negative_paths))]:
    #         data_unbalanced_val.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[int(0.15*len(silence_paths)):int(0.3*len(silence_paths))]:
    #         data_unbalanced_val.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     # Test set
    #     for path in positive_paths[:int(0.15*len(positive_paths))]:
    #         data_unbalanced_test.append("positive" + " " + str(2.0) + " " + path + "\n")

    #     for path in negative_paths[:int(0.15*len(negative_paths))]:
    #         data_unbalanced_test.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     for path in silence_paths[:int(0.15*len(silence_paths))]:
    #         data_unbalanced_test.append("negative" + " " + str(2.0) + " " + path + "\n")

    #     random.shuffle(data_unbalanced_train)
    #     random.shuffle(data_unbalanced_val)
    #     random.shuffle(data_unbalanced_test)

    #     with open("data/trainsets/trainset4_ftel4_unbalanced/label_time_wav_70.txt", "w") as f:
    #         for line in data_unbalanced_train:
    #             f.write(line)
    #     with open("data/trainsets/trainset4_ftel4_unbalanced/label_time_wav_val_15.txt", "w") as f:
    #         for line in data_unbalanced_val:
    #             f.write(line)
    #     with open("data/trainsets/trainset4_ftel4_unbalanced/label_time_wav_test_15.txt", "w") as f:
    #         for line in data_unbalanced_test:
    #             f.write(line)

    # Create full test set on Ftel5
    positive_3s = glob("ftel5_5Dec2023/positive_3s/*.wav")
    silence_3s = glob("ftel5_5Dec2023/silence_3s/*.wav")
    positive_3s = [
        "positive" + " 2.0 " + positive_path + "\n" for positive_path in positive_3s
    ]
    # silence_3s = ["negative" + " 2.0 " + silence_path + "\n" for silence_path in silence_3s]

    # data_test = positive_3s + silence_3s
    random.shuffle(positive_3s)
    number_of_samples = 300
    for i in range(0, 10):
        with open(
            f"/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train_part_{str(i+1)}.txt",
            "w",
        ) as f:
            for j in range(number_of_samples):
                f.write(positive_3s[number_of_samples * i + j])
    i += 1
    with open(
        f"/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train_part_{str(i+1)}.txt",
        "w",
    ) as f:
        for positive_path in positive_3s[number_of_samples * i :]:
            f.write(positive_path)

################################################################################################
# ids = []
# for positive_path in positive_3s:
#     id = re.search(r"_\d+_", positive_path).group().split("_")[1]
#     ids.append(id)

# isolated_ids = [ids[random.randint(0, 3111)] for i in range(32)]
# count = 0
# train_isolated = []
# test_isolated = []
# for positive_path in positive_3s:
#     id = re.search(r"_\d+_", positive_path).group().split("_")[1]
#     if id in isolated_ids:
#         count += 1
#         test_isolated.append(positive_path)
#     else:
#         train_isolated.append(positive_path)

# print(count)

# with open("data/trainsets/trainset6_ftel5/train_isolated.txt", "w") as f:
#     for line in train_isolated:
#         f.write(line)

# with open("data/trainsets/trainset6_ftel5/test_isolated.txt", "w") as f:
#     for line in test_isolated:
#         f.write(line)

###############################################################################################
#     silence_3s = glob("ftel5_5Dec2023/silence_3s/*.wav")
#     silence_3s = ["negative" + " 2.0 " + silence_path + "\n" for silence_path in silence_3s]
#     txt_files = glob("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/data_train*.txt")
#     txt_files_test = [txt_files[random.randint(0, 10)] for i in range(3)]
#     data_train = []
#     data_test = []
#     for txt_file in txt_files:
#         if txt_file in txt_files_test:
#             with open(txt_file, "r") as f:
#                 lines = f.readlines()
#             data_test += lines

#         else:
#             with open(txt_file, "r") as f:
#                 lines = f.readlines()
#             data_train += lines

#     print(len(data_train))
#     print(len(data_test))

#     for silence_path in silence_3s[:int(len(silence_3s)*0.2)]:
#         data_test.append(silence_path)

#     for silence_path in silence_3s[int(len(silence_3s)*0.2):]:
#         data_train.append(silence_path)

#     random.shuffle(data_train)
#     random.shuffle(data_test)

#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/train_random10.txt", "w") as f:
#         for line in data_train:
#             f.write(line)
#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/test_random10.txt", "w") as f:
#         for line in data_test:
#             f.write(line)
# ##################################################################################################
#     silence_3s = glob("ftel5_5Dec2023/silence_3s/*.wav")
#     silence_3s = ["negative" + " 2.0 " + silence_path + "\n" for silence_path in silence_3s]

#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/train_isolated.txt", "r") as f:
#         data_train = f.readlines()

#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/test_isolated.txt") as f:
#         data_test = f.readlines()

#     print(len(data_train))
#     print(len(data_test))


#     for silence_path in silence_3s[:int(len(silence_3s)*0.2)]:
#         data_test.append(silence_path)

#     for silence_path in silence_3s[int(len(silence_3s)*0.2):]:
#         data_train.append(silence_path)

#     random.shuffle(data_train)
#     random.shuffle(data_test)

#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/train_isolated_add_silence10.txt", "w") as f:
#         for line in data_train:
#             f.write(line)
#     with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/test_isolated_add_silence10.txt", "w") as f:
#         for line in data_test:
#             f.write(line)
