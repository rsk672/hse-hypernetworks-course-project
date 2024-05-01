import torch
from torch import nn
import torch.nn.functional as F
from src.hnet_lib.hnet import HyperNetwork


class HyperLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

    def get_params_count(self):
        cnt = self.input_dim * self.output_dim
        
        if self.bias:
            cnt += self.output_dim

        return cnt

    def forward(self, x, hypernetwork_weights):
        w = hypernetwork_weights[:self.input_dim*self.output_dim].view(self.output_dim, self.input_dim)
        b = torch.zeros((self.output_dim, ))
        
        if self.bias:
            b = hypernetwork_weights[self.input_dim*self.output_dim:]

        return F.linear(x, w, b)


class HyperConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def get_params_count(self):
        cnt = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size
        
        if self.bias:
            cnt += self.out_channels
        return cnt

    def forward(self, x, hypernetwork_weights):
        conv_weights_cnt = self.in_channels*self.out_channels*self.kernel_size*self.kernel_size

        conv_weights = hypernetwork_weights[:conv_weights_cnt].view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        conv_bias = torch.zeros(self.out_channels)

        if self.bias:
            conv_bias = hypernetwork_weights[conv_weights_cnt:conv_weights_cnt+self.out_channels].view((self.out_channels,))

        return F.conv2d(x, conv_weights, conv_bias, stride=self.stride, padding=self.padding)


class HyperBatchNorm2d(nn.Module):
    def __init__(self, num_features, freezable=True, device='cpu'):
        super().__init__()

        self.num_features = num_features
        self.running_mean = torch.zeros(num_features).to(device)
        self.running_var = torch.ones(num_features).to(device)
        self.hnet = HyperNetwork(5, 10, 2 * num_features).to(device)
        
        self.freezable = freezable
        self.update_running_stats = True
    
    def forward(self, x, task_embedding):
        affine_weights = self.hnet.get_task_weights(task_embedding)

        w = affine_weights[:self.num_features]
        b = affine_weights[self.num_features:2*self.num_features]
        
        if self.training:
            for p in self.hnet.parameters():
                p.requires_grad = True
            if not self.freezable:
                self.update_running_stats = True
        else:
            self.update_running_stats = False
        
        return F.batch_norm(x, self.running_mean, self.running_var, w, b, training=self.update_running_stats)