"""
A minimal script for generating 4 moving numbers as video.
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image
import imageio

def video_horizon(img, pad_unit, pad_cnt):
    """
      video horizon
    """
    img_size = img.size[0]
    new_size = img_size + pad_cnt * pad_unit
    hor_imgs = []
    for i in range(4):
        new_img = Image.new("L", (new_size, new_size), 0)
        new_img.paste(img, (pad_unit // 2 + pad_unit *
                            i, pad_cnt * pad_unit // 2))
        hor_imgs.append(new_img)

    return hor_imgs


def video_vert(img, pad_unit, pad_cnt):
    """
      video vert
    """
    img_size = img.size[0]
    new_size = img_size + pad_cnt * pad_unit
    ver_imgs = []
    for i in range(4):
        new_img = Image.new("L", (new_size, new_size), 0)
        new_img.paste(img, (pad_cnt * pad_unit // 2, pad_unit // 2 + pad_unit *
                            i))
        ver_imgs.append(new_img)

    return ver_imgs


def video_diag_l(img, pad_unit, pad_cnt):
    """
      video diag left
    """
    img_size = img.size[0]
    new_size = img_size + pad_cnt * pad_unit
    diag_l_imgs = []
    for i in range(4):
        new_img = Image.new("L", (new_size, new_size), 0)
        new_img.paste(img, (pad_unit // 2 + pad_unit *
                            i, pad_unit // 2 + pad_unit *
                            i))
        diag_l_imgs.append(new_img)

    return diag_l_imgs


def video_diag_r(img, pad_unit, pad_cnt):
    """
      video diag right
    """
    img_size = img.size[0]
    new_size = img_size + pad_cnt * pad_unit
    diag_r_imgs = []
    for i in range(4):
        new_img = Image.new("L", (new_size, new_size), 0)
        new_img.paste(img, (new_size - (pad_unit // 2 + pad_unit *
                            i + img_size + 1), pad_unit // 2 + pad_unit *
                            i))
        diag_r_imgs.append(new_img)

    return diag_r_imgs

def video_static(img, pad_unit, pad_cnt):
    """
        video static
    """
    img_size = img.size[0]
    new_size = img_size + pad_cnt * pad_unit
    new_img = Image.new("L", (new_size, new_size), 0)
    new_img.paste(img, (pad_unit * pad_cnt // 2,  pad_unit * pad_cnt // 2))
    static_imgs = [new_img] * 4

    return static_imgs


def main(args):
    os.makedirs(args.video_path, exist_ok=True)
    for i in range(10):
        os.makedirs(os.path.join(args.video_path,
                    f'{i}'), exist_ok=True)

    # 32 + 5 * 3 = 47    ----> 48
    # 48 * 2 = 64
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size))
    ])

    dataset = MNIST("./data", train=True, download=True, transform=transform)
    dataset_size = len(dataset)
    pad_unit = 1
    pad_cnt = 4
    videos = [[] for i in range(10)]
    print(" ===============> video ,,, ")
    for i in tqdm(range(dataset_size)):
        img, label = dataset[i]
        # hor
        hor_imgs = video_horizon(img, pad_unit, pad_cnt)
        videos[label].append(hor_imgs)
        videos[label].append(hor_imgs[::-1])

        # ver
        ver_imgs = video_vert(img, pad_unit, pad_cnt)
        videos[label].append(ver_imgs)
        videos[label].append(ver_imgs[::-1])

        # diag_l
        diag_l_imgs = video_diag_l(img, pad_unit, pad_cnt)
        videos[label].append(diag_l_imgs)
        videos[label].append(diag_l_imgs[::-1])

        # diag_r
        diag_r_imgs = video_diag_r(img, pad_unit, pad_cnt)
        videos[label].append(diag_r_imgs)
        videos[label].append(diag_r_imgs[::-1])

        # static
        static_imgs = video_static(img, pad_unit, pad_cnt)
        videos[label].append(static_imgs)



    print(" ===============> video done. ")
    print(" ===============> save videos ,,, ")
    
    for i in tqdm(range(10)):
        for j in tqdm(range(len(videos[i]))):
            with imageio.get_writer(f'{args.video_path}/{i}/{j}.mp4', fps=0.5) as v:
                imgs = videos[i][j]
                for img in imgs:
                    img = np.array(img, dtype=np.uint8)
                    v.append_data(img)

    print(" ===============> save done. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default="video")
    parser.add_argument("--image-size", type=int, default=28)
    args = parser.parse_args()
    main(args)