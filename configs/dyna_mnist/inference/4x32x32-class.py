num_frames = 4
fps = 1
image_size = (32, 32)

# Define model
model = dict(
    type="Latte-S/2",
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
batch_size = 2
seed = 42
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./outputs/samples/"
