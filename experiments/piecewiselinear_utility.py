import torch


class PiecewiseLinear(torch.nn.Module):
    def __init__(self, beta1, beta2, thresholds):
        super().__init__()
        self.register_buffer("beta1", beta1)
        self.register_buffer("beta2", beta2)
        self.register_buffer("thresholds", thresholds)

    def calc_raw_util_per_dim(self, Y):
        # below thresholds
        bt = Y < self.thresholds
        b1 = self.beta1.expand(Y.shape)
        b2 = self.beta2.expand(Y.shape)
        shift = (b2 - b1) * self.thresholds
        util_val = torch.empty_like(Y)

        # util_val[bt] = Y[bt] * b1[bt]
        util_val[bt] = Y[bt] * b1[bt] + shift[bt]
        util_val[~bt] = Y[~bt] * b2[~bt]

        return util_val

    def forward(self, Y, X=None):
        util_val = self.calc_raw_util_per_dim(Y)
        util_val = util_val.sum(dim=-1)
        return util_val