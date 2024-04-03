import argparse
import csv
import os
import random

from torchvision.datasets import ImageNet


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


def split_by_capital(name):
    # BoxingPunchingBag -> Boxing Punching Bag
    new_name = ""
    for i in range(len(name)):
        if name[i].isupper() and i != 0:
            new_name += " "
        new_name += name[i]
    return new_name


def process_imagenet(root, split):
    root = os.path.expanduser(root)
    data = ImageNet(root, split=split)
    samples = [(path, data.classes[label][0]) for path, label in data.samples]
    output = f"imagenet_{split}.csv"

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


def process_ucf101(root, split):
    root = os.path.expanduser(root)
    video_lists = get_filelist(os.path.join(root, split))
    classes = [x.split("/")[-2] for x in video_lists]
    classes = [split_by_capital(x) for x in classes]
    samples = list(zip(video_lists, classes))
    output = f"ucf101_{split}.csv"

    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


def process_dyna_mnist(root, text):
    """
      note: 
      python3 convert_dataset.py dyna_mnist ~/Downloads/data/DynaMNIST_20240321
    """
    root = os.path.expanduser(root)
    video_lists = get_filelist(root)
    # print(video_lists[0])
    nums = [int(x.split("/")[-2]) for x in video_lists]
    # print(set(classes))
    words = ["Zero", "One", "Two", "Three", "Four",
             "Five", "Six", "Seven", "Eight", "Nine"]
    classes_text = [words[x] for x in nums]
    classes = classes_text if text else nums
    samples = list(zip(video_lists, classes))
    
    # note: random sample 1 of 8
    random_samples = []
    if True:
        total_size = len(samples)
        ind_lst = list(range(0, total_size, 8))
        for i in ind_lst:
            ind = i + random.randint(0, 8)
            random_samples.append(samples[ind])

    output = f"DynaMNIST.csv"
    with open(output, "w") as f:
        writer = csv.writer(f)
        if True:
            writer.writerows(random_samples)
        else:
            writer.writerows(samples)

    print(f"Saved {len(samples)} samples to {output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=[
                        "imagenet", "ucf101", "dyna_mnist"])
    parser.add_argument("root", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text", type=bool, default=False)
    args = parser.parse_args()

    if args.dataset == "imagenet":
        process_imagenet(args.root, args.split)
    elif args.dataset == "ucf101":
        process_ucf101(args.root, args.split)
    elif args.dataset == "dyna_mnist":
        process_dyna_mnist(args.root, args.text)
    else:
        raise ValueError("Invalid dataset")
