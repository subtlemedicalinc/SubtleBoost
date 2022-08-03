import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def cyclic_shift(x, shifts, dims):
    #x: B, n_contrast, H, W, C
    n = x.shape[1]
    shifted_x = []
    for i in range(n):
        shifted_x_i = torch.roll(x[:, i, :, :, :], shifts=shifts, dims=dims)
        shifted_x.append(shifted_x_i.unsqueeze(1))
    shifted_x = torch.cat(shifted_x, 1)
    return shifted_x




def window_partition(x, window_size):
    """
    Args:
        x: (B, n_contrast, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, n_contrast, H, W, C = x.shape
    windows = []
    for i in range(n_contrast):
        x_i = x[:, i, :, :, :]
        x_i = x_i.view(B, H // window_size, window_size, W // window_size, window_size, C)
        window_i = x_i.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, 1, window_size, window_size, C)  #(num_windows*B, 1, window_size, window_size, C)
        windows.append(window_i)
    windows = torch.cat(windows, 1) #(num_windows*B, n_contrast, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, n_contrast, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, n_contrast, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = []
    n_contrast = windows.shape[1]
    for i in range(n_contrast):
        window = windows[:, i, :, :, :]
        x_i = window.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x_i = x_i.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 1, H, W, -1)
        x.append(x_i)
    x = torch.cat(x, 1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, n_contrast, window_size, window_size, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, n_contrast_q, Wh, Ww, C = x_q.shape
        n_contrast_kv = x_k.shape[1]

        x_q = x_q.view(-1, Wh * Ww * n_contrast_q, C)   #B*nW, Wh * Ww * n_contrast_q, C
        x_k = x_k.view(-1, Wh * Ww * n_contrast_kv, C)
        x_v = x_v.view(-1, Wh * Ww * n_contrast_kv, C)

        B_, N_q, C = x_q.shape
        _, N_kv, C = x_k.shape

        q = self.q(x_q).reshape(B_, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = self.k(x_k).reshape(B_, N_kv, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = self.v(x_v).reshape(B_, N_kv, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        # (1, B_, nH, N, C//n_head)
        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, nH, N_q, N_kv)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        # pdb.set_trace()   # check if the shape of relative position bias matches attn
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().repeat(1, n_contrast_q, n_contrast_kv)   # nH, Wh*Ww*c_q, Wh*Ww*c_kv
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: # nW, window_size**2, window_size**2
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.repeat(1, n_contrast_q, n_contrast_kv).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)  #B_, N_q, nH, c => B_, N_q, C
        x = self.proj(x)
        x = self.proj_drop(x).view(B_, n_contrast_q, Wh, Ww, C)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerEncoderBlock(nn.Module):
    r""" Swin Transformer Encoder Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask.unsqueeze(1), self.window_size)  # nW, 1, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size**2, window_size**2
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def with_pos_embed(self, tensor, pos=None):
        if pos is not None:
            n_contrast = pos.shape[0]
            return tensor + pos.view(1, n_contrast, 1, 1, -1)
        return tensor

    def forward(self, x, contrast_embed=None):
        # x: input (query); x_kv: decoder outputs
        # H, W = self.input_resolution
        B, n_contrast, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        x_windows_embedded = self.with_pos_embed(x_windows, contrast_embed)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*n_contrast, C

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size, self.window_size, C) # nW*B, n_contrast, window_size, window_size, C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = cyclic_shift(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))   # B, n_contrast, H, W, C

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class SwinTransformerDecoderBlock(nn.Module):
    r""" Swin Transformer Decoder Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.self_attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask.unsqueeze(1), self.window_size)  # nW, 1, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size**2, window_size**2
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def with_pos_embed(self, tensor, pos=None):
        if pos is not None:
            n_contrast = pos.shape[0]
            return tensor + pos.view(1, n_contrast, 1, 1, -1)
        return tensor

    def forward(self, x, x_kv, contrast_embed=None, contrast_token=None):
        # x: input (query); x_kv: decoder outputs
        # H, W = self.input_resolution
        B, n_contrast, H, W, C = x.shape


        ###### self attn
        shortcut = x
        x = self.norm1(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        # W-MSA/SW-MSA
        x_windows_embedded = self.with_pos_embed(x_windows, contrast_token)
        attn_windows = self.self_attn(x_q=x_windows_embedded, x_k=x_windows_embedded, x_v=x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*n_contrast, C

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size, self.window_size, C) # nW*B, n_contrast, window_size, window_size, C


        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = cyclic_shift(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)


        ##### cross attn
        shortcut = x
        x = self.norm2(x)  # B, n_contrast, H, W, C

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = cyclic_shift(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_x_kv = cyclic_shift(x_kv, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            shifted_x_kv = x_kv

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        x_kv_windows = window_partition(shifted_x_kv, self.window_size)  # nW*B, n_contrast, window_size, window_size, C
        # W-MSA/SW-MSA

        attn_windows = self.cross_attn(x_q=self.with_pos_embed(x_windows, contrast_token),
                                       x_k=self.with_pos_embed(x_kv_windows, contrast_embed),
                                       x_v=x_kv_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*n_contrast, C

        # merge windows
        attn_windows = attn_windows.view(-1, n_contrast, self.window_size, self.window_size, C) # nW*B, n_contrast, window_size, window_size, C

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B n_contrast H' W'  C
        # reverse cyclic shift
        if self.shift_size > 0:
            x = cyclic_shift(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)


        ##### FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))   # B, n_contrast, H, W, C

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        #H, W = self.input_resolution
        B, n_contrast, H, W, C = x.shape
        #assert L == H * W * n_contrast, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_merged = []
        for i in range(n_contrast):
            x0 = x[:, i, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, i, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, i, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, i, 1::2, 1::2, :]  # B H/2 W/2 C
            x_merged_i = torch.cat([x0, x1, x2, x3], -1).unsqueeze(1) # B 1 H/2 W/2 4*C
            x_merged.append(x_merged_i)
        x_merged = torch.cat(x_merged, 1) # B n_contrast H/2 W/2  4*C

        x = self.norm(x_merged)   # B n_contrast H/2 W/2  4*C
        x = self.reduction(x)  # B n_contrast H/2 W/2  2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        B, n_contrast, H, W, C = x.shape
        x = self.expand(x)    #B, n_contrast, H, W, 2C

        x = rearrange(x, 'b n h w (p1 p2 c)-> b n (h p1) (w p2) c', n=n_contrast, p1=2, p2=2, c=C//2)
        x= self.norm(x)   #B, n_contrast, 2*H, 2*W, C//2

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, n_contrast, H, W, C
        """
        # x = self.expand(x)
        n_contrast = x.shape[1]
        C = x.shape[-1]
        x = rearrange(x, 'b n h w (p1 p2 c)-> b n (h p1) (w p2) c', n=n_contrast,
                      p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = self.norm(x)  # B, n_contrast, H*dim_scale, W*dim_scale, C//dim_scale**2

        return x

class EncoderLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample((input_resolution[0]*2, input_resolution[1]*2), dim=dim//2, norm_layer=norm_layer)
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerEncoderBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])



    def forward(self, x, contrast_embed=None):
        if self.downsample is not None:
            x = self.downsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, contrast_embed)
            else:
                x = blk(x, contrast_embed)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class DecoderLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand((input_resolution[0]//2, input_resolution[1]//2), dim=dim*2, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerDecoderBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])


    def forward(self, x, x_kv, contrast_embed=None, contrast_token=None):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_kv, contrast_embed, contrast_token)
            else:
                x = blk(x, x_kv, contrast_embed, contrast_token)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)  # B Ph Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MMT(nn.Module):
    r""" MMT Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=1,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, ce=True,
                 use_checkpoint=False, final_upsample="expand_first", num_contrast=4, **kwargs):
        super().__init__()

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{}".format(depths,
        depths_decoder,drop_path_rate))

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.ce = ce  # add contrast embeddings
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.n_contrast = num_contrast

        # split image into non-overlapping patches
        # self.patch_embeds = nn.ModuleList([PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None) for _ in range(num_contrast)])  # different linear emb

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                 norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.contrast_embeds = nn.ModuleList([nn.Embedding(num_contrast, int(embed_dim * 2 ** i))
                                              for i in range(self.num_layers)])
        self.contrast_tokens = nn.ModuleList([nn.Embedding(num_contrast, int(embed_dim * 2 ** i))
                                              for i in range(self.num_layers)])

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers_enc = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer > 0)  else None,
                               use_checkpoint=use_checkpoint)
            self.layers_enc.append(layer)
        
        # build decoder layers
        self.layers_dec = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DecoderLayer(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                 input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                   patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                 depth=depths[(self.num_layers - 1 - i_layer)],
                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                     depths[:(self.num_layers - 1 - i_layer) + 1])],
                                 norm_layer=norm_layer,
                                 upsample=PatchExpand if (i_layer > 0) else None,
                                 use_checkpoint=use_checkpoint)
            self.layers_dec.append(layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size, img_size//patch_size), dim_scale=4,
                                          dim=embed_dim//patch_size**2)
            self.output = nn.Sequential(nn.Conv2d(in_channels=embed_dim//patch_size**2, out_channels=self.out_chans,kernel_size=1,bias=False),
                                        nn.ReLU())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder
    def forward_encoder(self, x, inputs, contrast_embeds=None):
        # initial embedding
        x_embed = []
        for contrast, x_i in zip(inputs, x):
            # x_i = self.patch_embeds[contrast](x_i).unsqueeze(1)  #B 1 Ph Pw C
            x_i = self.patch_embed(x_i).unsqueeze(1)  # B 1 Ph Pw C
            x_i = self.pos_drop(x_i)
            x_embed.append(x_i)
        x = torch.cat(x_embed, 1)  #B n_contrast Ph Pw C

        # transformer encoder processing
        enc_out = []

        for contrast_embed, layer in zip(contrast_embeds, self.layers_enc):
            x = layer(x, contrast_embed=contrast_embed)
            enc_out.append(x)
  
        return enc_out

    #Decoder
    def forward_decoder(self, x, enc_out, contrast_embeds=None, contrast_tokens=None):
        for i, layer_dec in enumerate(self.layers_dec):
            x = layer_dec(x, enc_out[self.num_layers-1-i], contrast_embed=contrast_embeds[self.num_layers-1-i],
                          contrast_token=contrast_tokens[self.num_layers-1-i])
        return x

    def up_x4(self, x):
        B, n_contrast, H, W, C = x.shape
        x = self.up(x)
        output = []
        for i in range(n_contrast):
            if self.final_upsample=="expand_first":
                x_i = x[:, i, :, :, :]
                x_i = self.output(x_i.permute(0, 3, 1, 2))
                output.append(x_i)
        return output

    def forward(self, x, inputs, outputs):
        contrast_embeds = [self.contrast_embeds[i].weight[inputs] for i in range(self.num_layers)]
        enc_out = self.forward_encoder(x, inputs, contrast_embeds=contrast_embeds)
        img_outputs = []
        for output in outputs:
            contrast_tokens = [self.contrast_tokens[i].weight[[output]] for i in range(self.num_layers)]
            B, _, H, W, C = enc_out[-1].shape
            tgt = torch.zeros(B, 1, H, W, C).cuda()
            x = self.forward_decoder(tgt, enc_out, contrast_embeds=contrast_embeds, contrast_tokens=contrast_tokens)
            x = self.up_x4(x)
            img_outputs.append(x[0])
        return img_outputs

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
