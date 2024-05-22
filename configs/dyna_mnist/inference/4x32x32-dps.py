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
dps_scale = 0.5
# super resolution downsample scale 8 -> dps scale 0.5; downsample scale 4 -> dps scale 0.1

