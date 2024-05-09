import torch
import torch.nn as nn
import numpy as np
from src.hnet_lib.hyper_layers import HyperLinear


class PatchEmbeddingLayer(nn.Module):

    def __init__(self, img_size, patch_size, input_channels=3, embedding_dim=48):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_conv = nn.Conv2d(
            in_channels=input_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.patch_conv(x).flatten(2).transpose(1, 2)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, mlp_hidden_size, n_heads=8, dropout=0., hyper_ffd=False, hyper_attention=False):
        super().__init__()

        self.ln1 = nn.LayerNorm(input_size)
        self.attn = MultiHeadSelfAttention(
            input_size, n_heads=n_heads, use_hnet=hyper_attention)
        self.ln2 = nn.LayerNorm(input_size)

        if hyper_ffd:
            self.mlp = nn.ModuleList([
                HyperLinear(input_size, mlp_hidden_size),
                nn.GELU(),
                HyperLinear(mlp_hidden_size, input_size),
                nn.GELU(),
            ])
        else:
            self.mlp = nn.ModuleList([
                nn.Linear(input_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, input_size),
                nn.GELU(),
            ])

    def get_weights_from_hnet_cnt(self):
        cnt = 0

        for m in self.mlp:
            if isinstance(m, HyperLinear):
                cnt += m.get_params_count()

        cnt += self.attn.get_weights_from_hnet_cnt()

        return cnt

    def forward(self, x, predicted_weights=None):
        used_weights_cnt = 0

        out = self.attn(self.ln1(
            x), predicted_weights[used_weights_cnt:used_weights_cnt+self.attn.get_weights_from_hnet_cnt()]) + x
        used_weights_cnt += self.attn.get_weights_from_hnet_cnt()

        skip_conn = out
        out = self.ln2(out)

        for m in self.mlp:
            if isinstance(m, HyperLinear):
                out = m(
                    out, predicted_weights[used_weights_cnt:used_weights_cnt+m.get_params_count()])
                used_weights_cnt += m.get_params_count()
            else:
                out = m(out)

        return out + skip_conn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, attn_bias=True, attn_dropout=0.01, mlp_dropout=0.01, use_hnet=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.head_dim = dim // n_heads

        self.scale = 1 / np.sqrt(self.head_dim)

        if use_hnet:
            self.attention_matrix = HyperLinear(dim, 3 * dim, attn_bias)
            self.mlp = HyperLinear(dim, dim)
        else:
            self.attention_matrix = nn.Linear(dim, 3 * dim, bias=attn_bias)
            self.mlp = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def get_weights_from_hnet_cnt(self):
        cnt = 0

        if isinstance(self.attention_matrix, HyperLinear):
            cnt += self.attention_matrix.get_params_count()

        if isinstance(self.mlp, HyperLinear):
            cnt += self.mlp.get_params_count()

        return cnt

    def forward(self, x, predicted_weights=None):

        batch_size, n_tokens, x_dim = x.shape
        used_weights_cnt = 0

        if isinstance(self.attention_matrix, HyperLinear):
            attention_matrix = self.attention_matrix(
                x, predicted_weights[used_weights_cnt:used_weights_cnt+self.attention_matrix.get_params_count()])
            used_weights_cnt += self.attention_matrix.get_params_count()
        else:
            attention_matrix = self.attention_matrix(x)

        attention_matrix = attention_matrix.reshape(
            batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        attention_matrix = attention_matrix.permute(2, 0, 3, 1, 4)

        Q, K, V = attention_matrix[0], attention_matrix[1], attention_matrix[2]

        K_T = K.transpose(-2, -1)
        attention = self.attn_dropout(
            (Q @ K_T / np.sqrt(self.head_dim)).softmax(dim=-1))
        values = (attention @ V).transpose(1, 2).flatten(2)

        if isinstance(self.mlp, HyperLinear):
            x = self.mlp(
                values, predicted_weights[used_weights_cnt:used_weights_cnt+self.mlp.get_params_count()])
        else:
            x = self.mlp(values)
        x = self.mlp_dropout(x)

        return x


class CustomViT(nn.Module):
    def __init__(self, num_classes=10, hyper_lp=False, hyper_ffd=False, hyper_attention=False):
        super().__init__()

        self.img_size = 32
        self.patches_count = 8
        self.dropout = 0.3
        self.num_encoders = 7
        self.hidden_size = 384
        self.mlp_hidden_size = 384
        self.n_heads = 8

        self.patch_size = self.img_size // self.patches_count
        n_tokens = (self.patches_count ** 2) + 1

        self.patch_embedding_layer = PatchEmbeddingLayer(
            img_size=self.img_size, patch_size=self.patch_size)

        self.linear_after_patch = nn.Linear(
            (self.img_size // self.patches_count) ** 2 * 3, self.hidden_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.positional_embeddings = nn.Parameter(
            torch.randn(1, n_tokens, self.hidden_size))

        self.encoders = nn.ModuleList([TransformerEncoder(input_size=self.hidden_size, mlp_hidden_size=self.mlp_hidden_size, dropout=self.dropout, n_heads=self.n_heads,
                                      hyper_ffd=hyper_ffd, hyper_attention=hyper_attention) for _ in range(self.num_encoders)])

        self.ln = nn.LayerNorm(self.hidden_size)

        if hyper_lp:
            self.head = HyperLinear(self.hidden_size, num_classes)
        else:
            self.head = nn.Linear(self.hidden_size, num_classes)

    def get_weights_from_hnet_cnt(self):
        cnt = 0

        if isinstance(self.head, HyperLinear):
            cnt += self.head.get_params_count()
        for enc in self.encoders:
            cnt += enc.get_weights_from_hnet_cnt()

        return cnt

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, predicted_weights):
        out = self.patch_embedding_layer(x)
        out = self.linear_after_patch(out)

        out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.positional_embeddings

        used_weights_cnt = 0

        for m in self.encoders:
            out = m(
                out, predicted_weights[used_weights_cnt:used_weights_cnt+m.get_weights_from_hnet_cnt()])
            used_weights_cnt += m.get_weights_from_hnet_cnt()

        out = out[:, 0]

        if isinstance(self.head, HyperLinear):
            out = self.head(out, predicted_weights[used_weights_cnt:])
        else:
            out = self.head(out)

        return out
