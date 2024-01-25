from glob import glob

if __name__ == "__main__":
    with open("data/backgound/all_bg_list.txt") as f:
        texts = f.readlines()

    paths = glob("waves/bg/youtube/wav/music/*.wav")
    for path in paths:
        texts.append(path + "\n")

    with open("data/backgound/all_bg_list_new.txt", "w") as f:
        f.writelines(texts)
