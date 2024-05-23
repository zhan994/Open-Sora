from functools import partial

import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


@SCHEDULERS.register_module("dps")
class DPS(SpacedDiffusion):
    def __init__(
            self,
            num_sampling_steps=None,
            timestep_respacing=None,
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=True,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
            cfg_scale=4.0,
    ):
        # step: 1 获取betas
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

        # step: 2 loss
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE

        # step: 3 timestep_respacing
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]

        # step: 4 SpacedDiffusion构建，默认1000steps，EPSILON，LEARNED_RANGE，MSE
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale

    def sample(
            self,
            model,
            text_encoder,
            ob,                                                          # note
            operator,                                                    # note
            dps_scale,                                                   # note
            z_size,
            prompts,
            device,
            additional_args=None,
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        # z = torch.cat([z, z], 0)                                       # note cancel cat
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        # model_args["y"] = torch.cat([model_args["y"], y_null], 0)      # note
        model_args["y"] = y_null

        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.p_sample_loop_progressive(                        # use p_sample_loop_progressive
            model,                                                       # use model directly
            z.shape,
            z,
            ob,                                                          # note
            operator,                                                    # note
            scale=dps_scale,                                             # note
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
        )
        print("observation size", ob.shape)
        # samples, _ = samples.chunk(2, dim=0)                           # note cancel chunk
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)