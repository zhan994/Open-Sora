import torchvision
from opensora.datasets import video_transforms
import numpy as np

vframes, aframes, info = torchvision.io.read_video(
    filename="/home/zhan/Downloads/data/DynaMNIST_20240321/0/0.mp4", pts_unit="sec", output_format="TCHW")
total_frames = len(vframes)
print(total_frames)
temporal_sample = video_transforms.TemporalRandomCrop(
    4)

# Sampling video frames
start_frame_ind, end_frame_ind = temporal_sample(total_frames)
frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, 4, dtype=int)
video = vframes[frame_indice]
print(video.shape)
