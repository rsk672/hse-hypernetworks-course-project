from torch import nn
import torch.nn.functional as F
from ..hnet_lib.hyper_layers import HyperBatchNorm2d, HyperConv2d


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, hyper_batch_norm_config='none', device='cpu'):
        super().__init__()

        use_hyper_batch_norm = hyper_batch_norm_config != 'none'
        batch_norm_freezable = hyper_batch_norm_config == 'freezable'

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
                out_channels),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
                out_channels),
            nn.Dropout(0.1)
        ])

        self.skip_connection = nn.ModuleList([nn.Identity()]) if in_channels == out_channels else nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                                                                                                 HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
                                                                                                                     out_channels),
                                                                                                                 nn.Dropout(0.1)])

    def forward(self, x, task_embedding):
        skip_conn = x
        for layer in self.skip_connection:
            if isinstance(layer, HyperBatchNorm2d):
                skip_conn = layer(skip_conn, task_embedding)
            else:
                skip_conn = layer(skip_conn)

        for layer in self.layers:
            if isinstance(layer, HyperBatchNorm2d):
                x = layer(x, task_embedding)
            else:
                x = layer(x)

        x = x + skip_conn
        return F.relu(x)


class HyperResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, hyper_batch_norm_config='none', device='cpu'):
        super().__init__()

        use_hyper_batch_norm = hyper_batch_norm_config != 'none'
        batch_norm_freezable = hyper_batch_norm_config == 'freezable'

        self.conv1 = HyperConv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = HyperConv2d(out_channels, out_channels, 3, 1, 1)

        self.has_skip_connection = False

        if in_channels != out_channels:
            self.has_skip_connection = True
            self.skip_conv = HyperConv2d(
                in_channels, out_channels, 3, stride, 1)
            self.bn_skip = HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
                out_channels),
            self.dropout_skip = nn.Dropout(0.1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride = stride

        self.bn1 = HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
            out_channels),
        self.dropout1 = nn.Dropout(0.1)
        self.bn2 = HyperBatchNorm2d(out_channels, batch_norm_freezable, device) if use_hyper_batch_norm else nn.BatchNorm2d(
            out_channels),
        self.dropout2 = nn.Dropout(0.1)

    def get_params_count(self):
        cnt = self.conv1.get_params_count() + self.conv2.get_params_count()

        if self.has_skip_connection:
            cnt += self.skip_conv.get_params_count()

        return cnt

    def get_conv_weights(self, hypernetwork_weights):
        conv_weights_cnt = self.in_channels * \
            self.out_channels * self.kernel_size * self.kernel_size
        conv_weights = hypernetwork_weights[:conv_weights_cnt]

        return conv_weights.view((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

    def forward(self, x, hypernetwork_weights, task_embedding):
        used_weights_cnt = 0

        first_conv_weights = hypernetwork_weights[:self.conv1.get_params_count(
        )]
        used_weights_cnt += self.conv1.get_params_count()
        second_conv_weights = hypernetwork_weights[used_weights_cnt:
                                                   used_weights_cnt+self.conv2.get_params_count()]
        used_weights_cnt += self.conv2.get_params_count()

        # Calculating skip connection part
        skip_conn = x
        if self.has_skip_connection:
            skip_conn_weights = hypernetwork_weights[used_weights_cnt:
                                                     used_weights_cnt+self.skip_conv.get_params_count()]
            skip_conn = self.skip_conv(skip_conn, skip_conn_weights)

            if isinstance(self.bn_skip, HyperBatchNorm2d):
                skip_conn = self.bn_skip(skip_conn, task_embedding)
            else:
                skip_conn = self.bn_skip(skip_conn)

            skip_conn = self.dropout_skip(skip_conn)

        # Applying first convolution:
        x = self.conv1(x, first_conv_weights)
        if isinstance(self.bn1, HyperBatchNorm2d):
            x = self.bn1(x, task_embedding)
        else:
            x = self.bn1(x)
        x = F.relu(self.dropout1(x))

        # Applying second convolution:
        x = self.conv2(x, second_conv_weights)
        if isinstance(self.bn2, HyperBatchNorm2d):
            x = self.bn2(x, task_embedding)
        else:
            x = self.bn2(x)

        # Adding skip connection to output
        return F.relu(x + skip_conn)
