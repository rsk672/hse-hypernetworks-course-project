import torch
from torch import nn
import torchvision
from torchvision.models import ViT_B_16_Weights
from src.hnet_lib.hyper_layers import HyperLinear


class PreTrainedViT(nn.Module):
    def __init__(self, num_classes=10, use_hnet=False):
        super().__init__()

        if use_hnet:
            self.head = HyperLinear(768, num_classes)
        else:
            self.head = nn.Linear(768, num_classes)

        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(
            weights=pretrained_vit_weights)

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

        self.vit_feature_extractor = nn.Sequential(
            *list(self.vit.children())[:-1])

    def _extract_vit_features(self, x):
        encoder = self.vit_feature_extractor[1]
        x = self.vit._process_input(x)
        cls_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = encoder(x)

        return x[:, 0]

    def get_weights_from_hnet_cnt(self):
        if isinstance(self.head, HyperLinear):
            return self.head.get_params_count()

        return 0

    def forward(self, x, predicted_weights=None):
        features = self._extract_vit_features(x)
        features.requires_grad = True

        if predicted_weights is not None:
            output = self.head(features, predicted_weights)
            print(f'{output.shape=}')
            return self.head(features, predicted_weights)

        return self.head(features)
