from torch import nn
import torch.nn.functional as F
from src.hnet_lib.hyper_layers import HyperConv2d, HyperBatchNorm2d, HyperLinear
from src.resnet.residual_blocks import HyperResidualBlock, ResidualBlock


class ResNet18(nn.Module):
    def __init__(self, num_classes, hyper_layers=[], hyper_batch_norm_config='none', device='cpu'):
        super().__init__()

        use_hyper_batch_norm = hyper_batch_norm_config != 'none'
        batch_norm_freezable = hyper_batch_norm_config == 'freezable'

        self.device = device

        if 0 in hyper_layers:
            self.init_conv = HyperConv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.init_conv = nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = HyperBatchNorm2d(
            64, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(64)

        self.conv_channels = [64, 64, 128, 256, 512]

        self.blocks = nn.ModuleList([])

        for i in range(1, 5):
            stride = 1 if i == 1 else 2
            if i in hyper_layers:
                self.blocks.append(HyperResidualBlock(
                    in_channels=self.conv_channels[i - 1], out_channels=self.conv_channels[i], stride=stride, hyper_batch_norm_config=hyper_batch_norm_config))
                self.blocks.append(HyperResidualBlock(
                    in_channels=self.conv_channels[i], out_channels=self.conv_channels[i], stride=1, hyper_batch_norm_config=hyper_batch_norm_config))
            else:
                self.blocks.append(ResidualBlock(
                    in_channels=self.conv_channels[i - 1], out_channels=self.conv_channels[i], stride=stride, hyper_batch_norm_config=hyper_batch_norm_config))
                self.blocks.append(ResidualBlock(
                    in_channels=self.conv_channels[i], out_channels=self.conv_channels[i], stride=1, hyper_batch_norm_config=hyper_batch_norm_config))

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        if 5 in hyper_layers:
            self.fc = HyperLinear(512, num_classes)
        else:
            self.fc = nn.Linear(512, num_classes)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_weights_from_hnet_cnt(self):
        cnt = 0

        if isinstance(self.init_conv, HyperConv2d):
            cnt += self.init_conv.get_params_count()

        for block in self.blocks:
            if isinstance(block, HyperResidualBlock):
                cnt += block.get_params_count()
            if isinstance(self.fc, HyperLinear):
                cnt += self.fc.get_params_count()

        return cnt

    def forward(self, x, w, task_embedding):
        used_weights_cnt = 0

        if isinstance(self.init_conv, HyperConv2d):
            weights_to_pass_cnt = self.init_conv.get_params_count()
            x = self.init_conv(
                x, w[used_weights_cnt:used_weights_cnt+weights_to_pass_cnt])
            used_weights_cnt += weights_to_pass_cnt
        else:
            x = self.init_conv(x)

        if isinstance(self.bn1, HyperBatchNorm2d):
            x = F.relu(self.bn1(x, task_embedding))
        else:
            x = F.relu(self.bn1(x))

        for i, block in enumerate(self.blocks):
            if isinstance(block, HyperResidualBlock):
                weights_to_pass_cnt = block.get_params_count()
                x = block(
                    x, w[used_weights_cnt:used_weights_cnt+weights_to_pass_cnt], task_embedding)
                used_weights_cnt += weights_to_pass_cnt
            else:
                x = block(x, task_embedding)

        x = self.avgpool(x)

        if isinstance(self.fc, HyperLinear):
            weights_to_pass_cnt = self.fc.get_params_count()
            x = self.fc(x.view(
                x.shape[0], -1), w[used_weights_cnt:used_weights_cnt+weights_to_pass_cnt])
            used_weights_cnt += weights_to_pass_cnt
        else:
            x = self.fc(x.view(x.shape[0], -1))

        return x
