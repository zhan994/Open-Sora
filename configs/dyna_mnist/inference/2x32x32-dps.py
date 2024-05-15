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
