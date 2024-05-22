# 1 inference 

```bash
torchrun --nproc_per_node 1 --standalone scripts/inference.py configs/dyna_mnist/inference/4x32x32-stdit2.py --ckpt-path pretrained_models/012-F4S1-STDiT-SS-2_20240513/epoch993-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt

```

```bash
num_frames = 4
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="STDiT-SS/2",
    condition="label_10",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="PRETRAINED_MODEL",
    enable_flashattn=False,
    enable_layernorm_kernel=False
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=10,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
batch_size = 1
seed = 0
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./output_samples/4x32x32-stdit2/"
```


# 2  inference_novae 2 frames 

```bash
torchrun --nproc_per_node 1 --standalone scripts/inference_novae.py configs/dyna_mnist/inference/2x32x32-class.py --ckpt-path pretrained_models/epoch330-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
```

```bash
num_frames = 2
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="Latte-XS/2",
    condition="label_10",
    from_pretrained="PRETRAINED_MODEL",
    enable_flashattn=False,
    enable_layernorm_kernel=False
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=10,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=100,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./output_samples/2x32x32-class/"
```

# 3  inference_dps_novae 2 frames 

```bash
torchrun --nproc_per_node 1 --standalone scripts/infer_dps_novae.py configs/dyna_mnist/inference/2x32x32-dps.py --ckpt-path pretrained_models/epoch330-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
```

```bash
num_frames = 2
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="Latte-XS/2",
    condition="label_10",
    from_pretrained="PRETRAINED_MODEL",
    enable_flashattn=False,
    enable_layernorm_kernel=False
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=10,
)
scheduler = dict(
    type="dps",
    num_sampling_steps=1000,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./output_samples/2x32x32-dps/"
ref_dir = "./reference/2x32x32-dps/"
dps_scale = 2

```

# 4 inference_dps_vae 4 frames 

```bash
 torchrun --nproc_per_node 1 --standalone scripts/infer_dps.py configs/dyna_mnist/inference/4x32x32-dps.py --ckpt-path pretrained_models/012-F4S1-STDiT-SS-2_20240513/epoch993-global_step310000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
```

```bash
num_frames = 2
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="Latte-XS/2",
    condition="label_10",
    from_pretrained="PRETRAINED_MODEL",
    enable_flashattn=False,
    enable_layernorm_kernel=False
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(num_frames = 4
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="STDiT-SS/2",
    condition="label_10",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained="PRETRAINED_MODEL",
    enable_flashattn=False,
    enable_layernorm_kernel=False
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="classes",
    num_classes=10,
)
scheduler = dict(
    type="dps",
    num_sampling_steps=1000,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
batch_size = 1
seed = 0
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./output_samples/4x32x32-dps/"
ref_dir = "./reference/4x32x32-dps/"
dps_scale = 0.8

```

