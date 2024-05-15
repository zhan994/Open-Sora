import os

import torch
import colossalai
import torch.distributed as dist
import torchvision

from mmengine.runner import set_random_seed
from opensora.datasets import get_transforms_video
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.util.resizer import Resizer
from colossalai.cluster import DistCoordinator

# pip install -v .
# torchrun --nproc_per_node 1 --standalone scripts/infer_dps.py configs/dyna_mnist/inference/4x32x32-dps.py --ckpt-path pretrained_models/012-F4S1-STDiT-SS-2_20240513/epoch993-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # step: 1 cfg & 分布式环境
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    if coordinator.world_size > 1:
        set_sequence_parallel_group(dist.group.WORLD)
        enable_sequence_parallelism = True
    else:
        enable_sequence_parallelism = False

    # ======================================================
    # step: 2 运行相关的参数
    # ======================================================
    torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # step: 3 build model & load weights
    # ======================================================
    # step: 3.1 构建模型
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(
        cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        # enable_sequence_parallelism=enable_sequence_parallelism, # note: latte cancel
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # step: 3.2 move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # step: 3.3 build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # step: 3.4 multi-resolution支持
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device,
                          dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]],
                          device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # step: 4 dps observation
    #         task: multi-frame super resolution
    # ======================================================
    ref_path = os.path.join(cfg.ref_dir, f"0.mp4")
    vframes, aframes, info = torchvision.io.read_video(
        filename=ref_path, pts_unit="sec", output_format="TCHW")
    print(aframes)
    print(info)
    transform = get_transforms_video(resolution=cfg.image_size[0])
    video = transform(vframes)
    video = video.transpose(0, 1)

    # def RGB_to_L(img_rgb):
    #     img_l = torch.empty(1, img_rgb.shape[1], img_rgb.shape[2], img_rgb.shape[3])
    #     img_l[0,:,:,:] = 0.299 * img_rgb[0,:,:,:] + 0.587 * img_rgb[1,:,:,:] + 0.114 * img_rgb[2,:,:,:]
    #     return img_l
    #
    # if in_channels==1:
    #     video = RGB_to_L(video)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path_ref = os.path.join(save_dir, f"sample_ref")
    save_sample(video, fps=info['video_fps'], save_path=save_path_ref)
    ob = video.reshape(1, video.shape[0], video.shape[1], video.shape[2], video.shape[3]).to(device)  # 1 C T H W
    print("ob raw shape", ob.shape)

    def down_sample_by_frame(ds_input, scale_factor):
        ds_in_shape = [ds_input.shape[0], ds_input.shape[1], ds_input.shape[3], ds_input.shape[4]]  # down sample in img N(1) C H W
        # print("down_sample_inshape", ds_in_shape)
        down_sample = Resizer(ds_in_shape, 1 / scale_factor).to(device)
        img_size = ds_input.shape[3]
        down_sample_size = img_size / scale_factor
        down_sample_size = int(down_sample_size)
        ds_output = torch.empty(ds_input.shape[0], ds_input.shape[1], ds_input.shape[2], down_sample_size, down_sample_size)
        for t in range(0, ds_input.shape[2]):
            ds_output[:, :, t, :, :] = down_sample(ds_input[:, :, t, :, :])
        return ds_output

    scale_factor = 8

    # dps operator
    def operator(x):
        # down_sample_novae = down_sample_by_frame(ref, scale_factor)
        decode_x = vae.decode(x.to(dtype))
        down_sample_vae = down_sample_by_frame(decode_x, scale_factor)
        return down_sample_vae

    ob = vae.encode(ob.to(dtype))
    ob = operator(ob).to(device)
    noise_level = 0.01
    ob = ob + torch.randn_like(ob, device=device) * noise_level
    ob_save = torch.squeeze(ob, dim=0)  # C T H W
    save_path_ob = os.path.join(save_dir, f"sample_ob_scale{scale_factor}")
    save_sample(ob_save, fps=info['video_fps'], save_path=save_path_ob)
    print("super resolution down sample scale", scale_factor)
    print("gaussian noise level", noise_level)
    print("observation shape", ob.shape)

    # # ======================================================
    # # step: 5 inference
    # # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size):   # len(prompts)
        batch_prompts = prompts[i: i + cfg.batch_size]
        samples = scheduler.sample(
            model,
            text_encoder,
            ob,
            operator,
            cfg.dps_scale,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        samples = vae.decode(samples.to(dtype))
        samples = samples.to(dtype)
        # print(samples[0,0,0,:,:])
        if coordinator.is_master():
            for idx, sample in enumerate(samples):
                print(f"Prompt: {batch_prompts[idx]}")
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                save_sample(sample, fps=cfg.fps, save_path=save_path)
                sample_idx += 1


if __name__ == "__main__":
    main()
