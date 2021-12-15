import copy
from collections import OrderedDict
import torch
import torch.nn as nn


class ModelEma:
    """Model Ema
    A moving average is kept of model weights and buffers.
    Note that for multiple gpu, ema must be defined after mode init,
    but before DataParallel.
    Args:
        model: nn.Layer, original modela with learnable params
        decay: float, decay rate for each update, default: 0.999
    """

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.module.to('cpu')
        self.decay = decay

    @torch.no_grad()
    def _update(self, model, update_fn):
        # update ema model parameters by model parameters
        # _处应返回name值，不需要
        for (_, ema_param), (_, model_param) in zip(
                self.module.named_parameters(), model.named_parameters()):
            ema_param.set_value(copy.deepcopy(update_fn(ema_param, model_param)))

        # update ema model buffers by model buffers
        for (_, ema_buf), (_, model_buf) in zip(
                self.module.named_buffers(), model.named_buffers()):
            ema_buf.set_value(copy.deepcopy(update_fn(ema_buf, model_buf)))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self):
        return self.module.state_dict()