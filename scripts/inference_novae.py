import os

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator

# pip install -v .
# torchrun --nproc_per_node 1 --standalone scripts/inference_novae.py configs/dyna_mnist/inference/2x32x32-class.py --ckpt-path pretrained_models/epoch330-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
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
    torch.set_grad_enabled(False)
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
        input_size=input_size,
        in_channels=1,
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
    # step: 4 inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i: i + cfg.batch_size]
        samples = scheduler.sample(
            model,
            text_encoder,
            z_size=(1, *input_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        # samples = vae.decode(samples.to(dtype))
        samples = samples.to(dtype)

        if coordinator.is_master():
            for idx, sample in enumerate(samples):
                print(f"Prompt: {batch_prompts[idx]}")
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                save_sample(sample, fps=cfg.fps, save_path=save_path)
                sample_idx += 1


if __name__ == "__main__":
    main()
