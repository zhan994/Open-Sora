from copy import deepcopy

import torch.nn as nn
from mmengine.registry import Registry


def build_module(module, builder, **kwargs):
    """
    构建模型
    Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    # step: 1 获取module的相关cfg
    if isinstance(module, dict):
        cfg = deepcopy(module)
        for k, v in kwargs.items():
            cfg[k] = v

        # step: 2 根据cfg构建模型
        # note: Registry根据MODELS.register_module("Latte-XL/2")
        return builder.build(cfg)
    elif isinstance(module, nn.Module):
        return module
    elif module is None:
        return None
    else:
        raise TypeError(
            f"Only support dict and nn.Module, but got {type(module)}.")


# note: Registry注册模型
MODELS = Registry(
    "model",
    locations=["opensora.models"],
)

SCHEDULERS = Registry(
    "scheduler",
    locations=["opensora.schedulers"],
)
