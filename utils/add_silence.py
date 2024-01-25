from glob import glob
import random

if __name__ == "__main__":
    # silence_paths = glob("waves/ftel4/silence/wav/*.wav")
    # negative_paths = glob("negative_3s/negative_3s_quangbd_data/*.wav")
    # filter_negative_paths = []
    # for path in negative_paths:
    #     if "volumn_normed" not in path:
    #         filter_negative_paths.append(path)
    # # positive_paths = glob("positive_3s/**/*.wav", recursive=True)
    # # positive123_paths = glob("positive123_3s/ftel123_clone_aligned/**/*.wav", recursive=True)
    # # positive_reverb = glob("augment_positive_reverb/*.wav")[:1333]
    # # print(len(positive_reverb))
    # positive_speedup = glob("augment_positive_speed_up/*.wav")
    # positive_slowdown = glob("augment_positive_slow_down/*.wav")
    # positive_pitch = glob("augment_positive_pitch/*.wav")
    # print("positive_speedup:", len(positive_speedup))
    # print("positive_slodown:", len(positive_slowdown))
    # print("positive_pitch:", len(positive_pitch))

    # # positive_speedup_1_11 = glob("augment_positive_speed_up_1_1.1/*.wav")[:2000]
    # # positive_slowdown_09_1 = glob("augment_positive_slow_down_0.9_1/*.wav")[:2000]

    # # positive_speedup_1_12 = glob("augment_positive_speed_up_1_1.2/*.wav")[:2000]
    # # positive_slowdown_08_1 = glob("augment_positive_slow_down_0.8_1/*.wav")[:2000]
    # # print(len(positive_speedup_1_12))
    # # print(len(positive_slowdown_08_1))

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/train75_shuffle.txt", "r") as f:
    #     train = f.readlines()

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/test15_shuffle.txt", "r") as f:
    #     test = f.readlines()

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val10_shuffle.txt", "r") as f:
    #     val = f.readlines()

    # # # add augment positive data to trainset
    # for path in positive_pitch:
    #     train.append(f"positive 2.0 {path}\n")

    # # print("silence_paths:", len(silence_paths))
    # # print("negative:", len(filter_negative_paths))
    # # print("reverb:", len(positive_reverb))
    # # print("train:", len(train))
    # # print("val:", len(val))
    # # print("test:", len(test))

    # for path in positive_speedup:
    #     train.append(f"positive 2.0 {path}\n")

    # for path in positive_slowdown:
    #     train.append(f"positive 2.0 {path}\n")

    # # # add positive batch 123 audios
    # # for path in positive123_paths[int(0.25*len(positive123_paths)):]:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive123_paths[int(0.15*len(positive123_paths)):int(0.25*len(positive123_paths))]:
    # #     val.append(f"positive 2.0 {path}\n")

    # # for path in positive123_paths[:int(0.15*len(positive123_paths))]:
    # #     test.append(f"positive 2.0 {path}\n")

    # # # add positive audios
    # # for path in positive_paths[int(0.25*len(positive_paths)):]:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_paths[int(0.15*len(positive_paths)):int(0.25*len(positive_paths))]:
    # #     val.append(f"positive 2.0 {path}\n")

    # # for path in positive_paths[:int(0.15*len(positive_paths))]:
    # #     test.append(f"positive 2.0 {path}\n")

    # # add silence audios
    # for path in silence_paths[int(0.25*len(silence_paths)):]:
    #     train.append(f"negative 2.0 {path}\n")

    # for path in silence_paths[int(0.15*len(silence_paths)):int(0.25*len(silence_paths))]:
    #     val.append(f"negative 2.0 {path}\n")

    # for path in silence_paths[:int(0.15*len(silence_paths))]:
    #     test.append(f"negative 2.0 {path}\n")

    # # add negative audios
    # for path in negative_paths[int(0.25*len(negative_paths)):int(0.5*len(negative_paths))]:
    #     train.append(f"negative 2.0 {path}\n")

    # for path in negative_paths[int(0.15*len(negative_paths)):int(0.25*len(negative_paths))]:
    #     val.append(f"negative 2.0 {path}\n")

    # for path in negative_paths[:int(0.15*len(negative_paths))]:
    #     test.append(f"negative 2.0 {path}\n")

    # print(len(train))
    # print(len(val))
    # print(len(test))

    # with open("data/trainsets/trainset6_ftel5/data_train_part_1.txt", "r") as f:
    #     positive_added1 = f.readlines()

    # with open("data/trainsets/trainset6_ftel5/data_train_part_2.txt", "r") as f:
    #     positive_added2 = f.readlines()

    # train += positive_added1
    # train += positive_added2
    # train += test

    # random.shuffle(train)
    # random.shuffle(val)
    # random.shuffle(test)

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/train_new_add_ftel5.txt", "w") as f:
    #     for line in train:
    #         f.write(line)

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/test_new_add_ftel5.txt", "w") as f:
    #     for line in test:
    #         f.write(line)

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val_new_add_ftel5.txt", "w") as f:
    #     for line in val:
    #         f.write(line)

    # # ##############################################################################################################################################################
    # with open("data/trainsets/trainset5_noisy_data/train_base.txt") as f:
    #     train = f.readlines()

    # # with open("data/trainsets/trainset5_noisy_data/positive_added.txt") as f:
    # #     positive_samples = f.readlines()

    # # train = train + positive_samples

    # # with open("data/trainsets/trainset5_noisy_data/val.txt") as f:
    # #     val = f.readlines()

    # # with open("data/trainsets/trainset5_noisy_data/test.txt") as f:
    # #     test = f.readlines()
    # # test[-1] = test[-1] + "\n"

    # print("train:", len(train))
    # # print("val:", len(val))
    # # print("test:", len(test))

    # # add silence audios
    # print("silence")
    # print(len(silence_paths))
    # # for path in silence_paths[int(0.2*len(silence_paths)):]:
    # #     train.append(f"negative 2.0 {path}\n")

    # # for path in silence_paths[int(0.1*len(silence_paths)):int(0.2*len(silence_paths))]:
    # #     val.append(f"negative 2.0 {path}\n")

    # # for path in silence_paths[:int(0.1*len(silence_paths))]:
    # #     test.append(f"negative 2.0 {path}\n")

    # print("train:", len(train))
    # # print("val:", len(val))
    # # print("test:", len(test))

    # # add negative audios
    # filter_negative_paths = filter_negative_paths[:6000]
    # print("negative")
    # print(len(filter_negative_paths))
    # # for path in filter_negative_paths[int(0.2*len(filter_negative_paths)):]:
    # #     train.append(f"negative 2.0 {path}\n")

    # # for path in filter_negative_paths[int(0.1*len(filter_negative_paths)):int(0.2*len(filter_negative_paths))]:
    # #     val.append(f"negative 2.0 {path}\n")

    # # for path in filter_negative_paths[:int(0.1*len(filter_negative_paths))]:
    # #     test.append(f"negative 2.0 {path}\n")

    # print("train:", len(train))
    # # print("val:", len(val))
    # # print("test:", len(test))

    # # for path in positive_reverb:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_speedup:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_slowdown:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_pitch:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_speedup_1_11:
    # #     train.append(f"positive 2.0 {path}\n")

    # # for path in positive_slowdown_09_1:
    # #     train.append(f"positive 2.0 {path}\n")

    # for path in positive_speedup_1_12:
    #     train.append(f"positive 2.0 {path}\n")

    # for path in positive_slowdown_08_1:
    #     train.append(f"positive 2.0 {path}\n")

    # print("train:", len(train))
    # # print("val:", len(val))
    # # print("test:", len(test))

    # random.shuffle(train)
    # # random.shuffle(val)
    # # random.shuffle(test)

    # with open("data/trainsets/trainset5_noisy_data/train_speedup_1_12_slow_down_08_1.txt", "w") as f:
    #     for line in train:
    #         f.write(line)

    # # with open("data/trainsets/trainset5_noisy_data/test_new.txt", "w") as f:
    # #     for line in test:
    # #         f.write(line)

    # # with open("data/trainsets/trainset5_noisy_data/val_new.txt", "w") as f:
    # #     for line in val:
    # #         f.write(line)
    ####################################################################################################################

    #     with open("data/trainsets/trainset5_noisy_data/train_speedup_slowdown.txt", "r") as f:
    #         train = f.readlines()

    #     with open("data/trainsets/trainset6_ftel5/data_train_part_1.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data = train + positive_added
    #     random.shuffle(data)

    #     with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_1part_positive.txt", "w") as f:
    #         for line in data:
    #             f.write(line)
    # #####################################################################################################################
    #     data = []
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_2.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_2part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_3.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_3part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_4.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_4part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_5.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_5part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_6.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_6part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_7.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_7part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_8.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_8part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_9.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_8part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_10.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     # with open("data/trainsets/trainset6_ftel5/train_speedup_slowdown_added_8part_positive.txt", "w") as f:
    #     #     for line in data:
    #     #         f.write(line)
    # # #####################################################################################################################
    #     with open("data/trainsets/trainset6_ftel5/data_train_part_11.txt", "r") as f:
    #         positive_added = f.readlines()

    #     data += positive_added
    #     random.shuffle(data)

    #     silence_3s = glob("ftel5_5Dec2023/silence_3s/*.wav")
    #     silence_3s = ["negative" + " 2.0 " + silence_path + "\n" for silence_path in silence_3s]

    #     negative = glob("ftel5_5Dec2023/databugs/*.wav")
    #     negative = ["negative" + " 2.0 " + negative_path + "\n" for negative_path in negative]
    #     # print(len(negative))

    #     with open("data/trainsets/trainset6_ftel5/test_10parts.txt", "w") as f:
    #         for line in data:
    #             f.write(line)

    #         for silence in silence_3s:
    #             f.write(silence)

    #         for negative_path in negative:
    #             f.write(negative_path)

    # with open("/tf/train_hiFPToi/data/trainsets/wuw_ds4a/val_new_add_ftel5.txt", "r") as f:
    #     val = f.readlines()

    # with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/test_10parts.txt", "r") as f:
    #     test = f.readlines()

    # val += test

    # with open("/tf/train_hiFPToi/data/trainsets/trainset6_ftel5/test_merged.txt", "w") as f:
    #     for line in val:
    #         f.write(line)
    with open(
        "/tf/train_hiFPToi/data/trainsets/wuw_ds4a/train_new_add_ftel5.txt", "r"
    ) as f:
        train = f.readlines()

    ftel5a_paths = glob("ftel5_5Dec2023/positive_3s/*.wav")
    ftel5a_paths = ["positive" + " 2.0 " + path + "\n" for path in ftel5a_paths]
    # print("1", len(ftel5a_paths))
    train += ftel5a_paths

    ftel5b_paths = glob("Allb_3s/*.wav")
    ftel5b_paths = ["positive" + " 2.0 " + path + "\n" for path in ftel5b_paths]
    # print("2", len(ftel5b_paths))
    train += ftel5b_paths

    ftel5_positive_slow_down = glob("ftel5_positive_slow_down/*.wav")
    ftel5_positive_slow_down = [
        "positive" + " 2.0 " + path + "\n" for path in ftel5_positive_slow_down
    ]
    # print("3", len(ftel5_positive_slow_down))
    train += ftel5_positive_slow_down

    ftel5_positive_speed_up = glob("ftel5_positive_speed_up/*.wav")
    ftel5_positive_speed_up = [
        "positive" + " 2.0 " + path + "\n" for path in ftel5_positive_speed_up
    ]
    # print("4", len(ftel5_positive_speed_up))
    train += ftel5_positive_speed_up

    ftel5_positive_pitch = glob("ftel5_positive_pitch/*.wav")
    ftel5_positive_pitch = [
        "positive" + " 2.0 " + path + "\n" for path in ftel5_positive_pitch
    ]
    # print("5", len(ftel5_positive_pitch))
    train += ftel5_positive_pitch

    random.shuffle(train)

    with open(
        "/tf/train_hiFPToi/data/trainsets/wuw_ds4a/train_new_add_fullftel5a_ftel5b_augmentftel5.txt",
        "w",
    ) as f:
        for line in train:
            f.write(line)
