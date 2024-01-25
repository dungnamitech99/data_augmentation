from glob import glob
import random

if __name__ == "__main__":
    extra_positive = glob(
        "analysis_results/dscnn_normed_layer_6_v1_miss_rate_audio/*.wav"
    )
    extra_negative = glob(
        "analysis_results/dscnn_normed_layer_6_v1_false_alarm_audio/*.wav"
    )

    with open(
        "/tf/train_hiFPToi/data/trainsets/wuw_ds4a/train75_mixed_negative_add_positive_volumn_normed_shuffle_latest.txt",
        "r",
    ) as f:
        train = f.readlines()

    print(train[:5])
    for positive_path in extra_positive:
        train.append(f"positive 2.0 {positive_path}\n")

    for negative_path in extra_negative:
        train.append(f"negative 2.0 {negative_path}\n")

    random.shuffle(train)

    with open(
        "/tf/train_hiFPToi/data/trainsets/wuw_ds4a/dscnn_train75_mixed_negative_add_positive_volumn_normed_shuffle_latest.txt",
        "w",
    ) as f:
        for line in train:
            f.write(line)

    # print(f"positve: {len(extra_positive)}")
    # print(f"negative: {len(extra_negative)}")
