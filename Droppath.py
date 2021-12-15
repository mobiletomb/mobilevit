import numpy as np
import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self,
                 drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x):
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        keep_prob = torch.tensor(keep_prob, dtype=torch.float32)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=torch.float32)
        random_tensor = random_tensor.floor() # mask
        output = x.div(keep_prob) * random_tensor

        return output

    def forward(self, x):
        return self.drop_path(x)


if __name__ == "__main__":
    a = torch.arange(8*16*8*8).view(8,16,8,8)
    dp = DropPath(0.5)
    print(dp(a))