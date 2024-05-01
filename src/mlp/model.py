from torch import nn
import torch.nn.functional as F
from ..hnet_lib.hyper_layers import HyperLinear

class MLP(nn.Module):
    def __init__(self, hyper_layers=[], device='cpu'):
        super().__init__()

        self.hyper_layers = hyper_layers
        self.layer_dims = [784, 100, 32, 10]

        self.layers = nn.ModuleList([])
        for i in range(len(self.layer_dims) - 1):
            if i in self.hyper_layers:
                self.layers.append(HyperLinear(self.layer_dims[i], self.layer_dims[i + 1]))
            else:
                self.layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))

    def freeze_finetune(self):
        for p in self.layers[0].parameters():
            p.requires_grad = False

        for p in self.layers[1].parameters():
            p.requires_grad = False

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_weights_from_hnet_cnt(self):
        cnt = 0

        for layer in self.layers:
            if isinstance(layer, HyperLinear):
                cnt += layer.get_params_count()

        return cnt

    def forward(self, x, w):
        used_weights_cnt = 0

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                weights_to_pass_cnt = layer.get_params_count()
                weights_to_pass = w[used_weights_cnt:used_weights_cnt+weights_to_pass_cnt]
                x = layer(x, weights_to_pass)
                used_weights_cnt += weights_to_pass_cnt

        if i < len(self.layers) - 1:
            x = F.relu(x)

        return x