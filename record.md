# 1 安装

针对`ubuntu2004`下的`python3.8 + cuda11.3 + pytorch1.12`安装。

```bash
pip3_install_pkg colossalai accelerate diffusers ftfy gdown mmengine pre-commit av tensorboard timm tqdm transformers wandb xformers==0.0.13 triton packaging ninja apex

# install flash attention (optional) cuda版本不够 
pip3_install_pkg flash-attn --no-build-isolation

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```

针对`ubuntu2004`下的`python3.10 + cuda11.8 + pytorch2.1.2`安装。

```bash
conda activate opensora
sudo /opt/anaconda3/envs/opensora/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple colossalai accelerate diffusers ftfy gdown mmengine pre-commit av tensorboard timm tqdm transformers wandb xformers==0.0.23.post1 triton

# install flash attention (optional)
pip3_install_pkg flash-attn --no-build-isolation

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```



# 2 训练推理

## 2.1 数据集

针对`DynaMNIST`数据使用以下脚本生成训练使用的`CSV`文件即可。

```bash
python3 tools/datasets/convert_dataset.py dyna_mnist ~/Downloads/data/DynaMNIST_20240321
```

## 2.2 训练

使用`4x32x32-class`的配置训练`DynaMNIST`，如下：

```bash
torchrun --nproc_per_node=1 --nnodes=1 scripts/train.py configs/dyna_mnist/train/4x32x32-class.py --data-path YOUR_CSV_PATH --load PRETRAINED_MODEL_DIR
```


```python
num_frames = 4
frame_interval = 1
image_size = (32, 32)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="Latte-S/2",
    condition="label_10",
    enable_flashattn=False,
    enable_layernorm_kernel=False,
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
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 5000
load = None

batch_size = 8
lr = 2e-5
grad_clip = 1.0
```

## 2.3 推理

使用`4x32x32-class`的配置采样`DynaMNIST`，如下：

```bash
torchrun --nproc_per_node 1 --standalone scripts/inference.py configs/dyna_mnist/inference/4x32x32-class.py --ckpt-path outputs/000-F4S1-Latte-S-2/epoch764-global_step179000/ema.pt --prompt-path assets/texts/dyna_mnist_id.txt
```

```python
num_frames = 4
fps = 0.5
image_size = (32, 32)

# Define model
model = dict(
    type="Latte-S/2",
    condition="label_10",
    from_pretrained="PRETRAINED_MODEL"
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
    type="dpm-solver",
    num_sampling_steps=20,
    cfg_scale=4.0,
)
dtype = "fp16"

# Others
batch_size = 2
seed = 42
prompt_path = "./assets/texts/dyna_mnist_id.txt"
save_dir = "./outputs/samples/"
```

