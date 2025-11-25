import torch
import torch.nn as nn
from torch.optim import Optimizer

G_discretize = torch.Generator().manual_seed(1337)

def is_discrete(x):
    # we should be able to losslessly roundtrip via an integer dtype
    return torch.all(torch.eq(x, x.round()))

def root_n_by_d(x):
    d = x.dim()
    n = x.numel()
    return n ** (1/d)

def discretize_multinomial(w, G, c=1, n=None):
    if n is None:
        # n = torch.round(torch.sqrt(torch.tensor(w.numel()))).long()
        n = torch.round(torch.tensor(root_n_by_d(w))).long()
    abs_w = torch.abs(w)
    sum_w = abs_w.sum()
    p = (abs_w + c) / (sum_w + w.numel() * c)
    x = torch.distributions.Multinomial(total_count=n.item(), probs=p).sample()
    return x * torch.sign(w)

class DiscreteGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return discretize_multinomial(grad_output, G_discretize)

class ZIMPerParamOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(ZIMPerParamOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if not is_discrete(p.data):
                    breakpoint()

                if p.grad is None:
                    continue

                # Example (remove this and implement your own logic):
                # p.data.add_(-group['lr'] * p.grad.data)
                p.data.add_(-discretize_multinomial(p.grad, G_discretize))

        return loss

class ZIMOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(ZIMOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Collect all gradients and their shapes
        all_grads = []
        param_shapes = []
        params_with_grads = []

        for group in self.param_groups:
            for p in group['params']:
                if not is_discrete(p.data):
                    breakpoint()

                if p.grad is None:
                    continue

                all_grads.append(p.grad.flatten())
                param_shapes.append(p.grad.shape)
                params_with_grads.append(p)

        if all_grads:
            # Single global multinomial over all parameters
            global_grad = torch.cat(all_grads)
            global_discretized = discretize_multinomial(global_grad, G_discretize)

            # Redistribute back to original parameter shapes
            start_idx = 0
            for p, shape in zip(params_with_grads, param_shapes):
                end_idx = start_idx + shape.numel()
                p_discretized = global_discretized[start_idx:end_idx].view(shape)
                p.data.add_(-p_discretized)
                start_idx = end_idx

        return loss

def custom_init(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = discretize_multinomial(param.data, G_discretize)
        elif 'bias' in name:
            torch.nn.init.zeros_(param.data)

        if not is_discrete(param.data):
            breakpoint()
