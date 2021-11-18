# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import configs
from .resnetv2 import ResNetV2
from .unet import UNet
import torch.nn.functional as F
import pdb


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_q, hidden_states_k, hidden_states_v):
        mixed_query_layer = self.query(hidden_states_q)
        mixed_key_layer = self.key(hidden_states_k)
        mixed_value_layer = self.value(hidden_states_v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class AttentionDecoder(nn.Module):
    def __init__(self, config, vis):
        super(AttentionDecoder, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        window_size = config.window_size
        self.window_size = window_size

        # define a parameter table of relative position bias
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.num_attention_heads, 1, self.window_size**2))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_q, hidden_states_k, hidden_states_v):
        mixed_query_layer = self.query(hidden_states_q)
        mixed_key_layer = self.key(hidden_states_k)
        mixed_value_layer = self.value(hidden_states_v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias.repeat(1, attention_scores.shape[2],
                                                                    attention_scores.shape[3] // self.window_size ** 2)
        attention_probs = self.softmax(attention_scores + relative_position_bias.unsqueeze(0))
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class AttentionRelativeBias(nn.Module):
    def __init__(self, config, vis):
        super(AttentionRelativeBias, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        window_size = config.window_size
        self.window_size = window_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.num_attention_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states_q, hidden_states_k, hidden_states_v):
        mixed_query_layer = self.query(hidden_states_q)
        mixed_key_layer = self.key(hidden_states_k)
        mixed_value_layer = self.value(hidden_states_v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        n_contrast = attention_scores.shape[2]//self.window_size**2
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().repeat(1, n_contrast, n_contrast)  # nH, Wh*Ww, Wh*Ww

        #pdb.set_trace()
        attention_probs = self.softmax(attention_scores + relative_position_bias.unsqueeze(0))
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights



class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, in_channels, patch_size, window_size, hidden_size, dropout=0):
        super(Embeddings, self).__init__()
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(dropout)
        self.window_size = window_size
        self.unfold = nn.Unfold(window_size, padding=window_size//2)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        pad = self.window_size // 2
        x = F.pad(x, (pad, pad, pad, pad))
        x = x.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)  # (B, hidden, n_patches^(1/2), n_patches^(1/2), window_size, window_size)
        x = x.flatten(4).flatten(2, 3)  # (B, hidden, n_patches, window_size**2)
        x = x.permute(0, 2, 3, 1)  # (B, n_patches, window_size**2, hidden)
        x = x.contiguous().view(-1, x.shape[-2], x.shape[-1])  # (B*n_patches, window_size**2, hidden)

        #embeddings = x + self.position_embeddings
        #embeddings = self.dropout(embeddings)
        # return embeddings
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(TransformerEncoderBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = AttentionRelativeBias(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class TransformerEncoder(nn.Module):
    def __init__(self, config, vis):
        super(TransformerEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["encoder_num_layers"]):
            layer = TransformerEncoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(TransformerDecoderBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.slf_attention_norm_ = LayerNorm(config.hidden_size, eps=1e-6)
        self.slf_attn = Attention(config, vis)
        self.enc_attention_norm_ = LayerNorm(config.hidden_size, eps=1e-6)
        self.enc_attn = AttentionDecoder(config, vis)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)


    def forward(self, x, enc_output):
        h = x
        x = self.slf_attention_norm_(x)
        x, weights = self.slf_attn(x, x, x)
        x = x + h

        h = x
        x = self.enc_attention_norm_(x)
        x, weights = self.enc_attn(x, enc_output, enc_output)
        x = x + h


        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class TransformerDecoder(nn.Module):
    def __init__(self, config, vis):
        super(TransformerDecoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.decoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["decoder_num_layers"]):
            layer = TransformerDecoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, enc_output):
        attn_weights = []
        for layer_block in self.layer:
            x, weights = layer_block(x, enc_output)
            if self.vis:
                attn_weights.append(weights)
        decoded = self.decoder_norm(x)
        return decoded, attn_weights


class DeEmbedding(nn.Module):
    def __init__(self, config, latent_size, img_size):
        super(DeEmbedding, self).__init__()
        self.fc = Linear(config.hidden_size, latent_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)
        self.fold = nn.Fold((img_size, img_size), config.patch_size, stride=config.patch_size)

    def forward(self, x, batch_size):
        # x: (B*n_patch, 1, hidden)
        x = x.view(batch_size, -1, x.shape[-1])  # (B, n_patch, hidden)
        x = self.fc(x)  # (B, n_patch, latent)
        x = x.permute(0, 2, 1)  # (B, latent, n_patch)
        x = self.fold(x)
        return x


class SubtleGeneratorUNet(nn.Module):
    def __init__(self, config, img_size=256, vis=False):
        super(SubtleGeneratorUNet, self).__init__()
        self.img_size = img_size
        self.n_contrast = config.n_contrast

        self.output_patches = config.output_patches
        self.input_patches = config.window_size ** 2

        self.latent_size = config.patch_size**2*config.input_channels
        self.embeddings = Embeddings(in_channels=config.input_channels, window_size=config.window_size,
                                     patch_size=config.patch_size, dropout=config.transformer["dropout_rate"],
                                     hidden_size=config.hidden_size)
        self.transformer_enc = TransformerEncoder(config, vis)
        self.transformer_dec = TransformerDecoder(config, vis)

        self.config = config
        self.contrast_token0 = nn.Parameter(torch.zeros(1, self.output_patches, config.hidden_size))
        self.contrast_token1 = nn.Parameter(torch.zeros(1, self.output_patches, config.hidden_size))
        self.contrast_token2 = nn.Parameter(torch.zeros(1, self.output_patches, config.hidden_size))
        self.contrast_token3 = nn.Parameter(torch.zeros(1, self.output_patches, config.hidden_size))

        self.contrast_embedding0 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.contrast_embedding1 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.contrast_embedding2 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.contrast_embedding3 = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.contrast_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, self.output_patches, config.hidden_size))
        #                                          for _ in range(config.n_contrast)])
        # self.contrast_embeddings = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        #                                          for _ in range(config.n_contrast)])

        self.deembedding = DeEmbedding(config, self.latent_size, img_size)

    def embed_imgs(self, img_codes, contrasts):
        embedding_outputs = []
        for contrast, img_code in zip(contrasts, img_codes):
            embedding_output = self.embeddings(img_code)  # (B*n_patch, window_size**2, hidden)
            # add contrast embeddings
            #embedding_output += self.contrast_embeddings[contrast]  #.repeat(embedding_output.shape[0], self.input_patches, 1)
            embedding_output += getattr(self, f'contrast_embedding{contrast}') #.repeat(embedding_output.shape[0], self.input_patches, 1)
            embedding_outputs.append(embedding_output)
        embeddings = torch.cat(embedding_outputs, 1)  # (B*n_patch, window_size**2*len(inputs), hidden)
        return embeddings

    def forward(self, img_list, inputs, targets):
        # img -> encoder -> embeddings -> transformer_enc -> transformer_dec -> decoder

        B = img_list[0].shape[0]
        embeddings = self.embed_imgs(img_list, inputs)

        # fuse latent codes
        enc_outputs, _ = self.transformer_enc(embeddings)

        img_outputs = []
        for target in targets:
            contrast_token = getattr(self, f'contrast_token{target}').repeat(enc_outputs.shape[0], 1, 1)

            dec_outputs, _ = self.transformer_dec(contrast_token, enc_outputs)   # (B*n_patch, self.output_patches, hidden)
            x = self.deembedding(dec_outputs, B)  # (B, img_size, img_size, encoder.output_channel)
            img_outputs.append(x)

        return img_outputs, None


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-L_8': configs.get_l8_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
    'ViT-UNet-B_16': configs.get_unet_b16_config(),
    'ViT-UNet-L_8': configs.get_unet_l8_config(),
    'ViT-UNet-L_16': configs.get_unet_l16_config()
}


