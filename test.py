from glob import glob

if __name__ == "__main__":
    positive_paths = glob("Allb_3s/*.wav")
    # print(len(positive_paths))
    with open("positive_record_test.txt", "w") as f:
        for positive_path in positive_paths:
            f.write(f"positive 2.0 {positive_path}\n")
