import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms


def _trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        high = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def _softmax_inplace(x, dim):
    torch.exp(x, out=x)
    x /= torch.sum(x, dim=dim, keepdim=True)
    return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        inplace_softmax=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.inplace_softmax = inplace_softmax
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(
            b, n, 3, self.num_heads, c // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        if self.inplace_softmax:
            attn *= self.scale
            attn = _softmax_inplace(attn, dim=-1)
        else:
            attn = (attn * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        inplace_softmax=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            inplace_softmax=inplace_softmax,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224,),
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_encoding(self, x, w, h):
        n_patch = x.shape[1] - 1
        n_ref = self.pos_embed.shape[1] - 1
        if n_patch == n_ref and w == h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, 0]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size + 0.1
        h0 = h // self.patch_embed.patch_size + 0.1
        side = math.sqrt(n_ref)
        patch_pos = nn.functional.interpolate(
            patch_pos.reshape(1, int(side), int(side), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / side, h0 / side),
            mode="bicubic",
        )
        assert int(w0) == patch_pos.shape[-2] and int(h0) == patch_pos.shape[-1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1)

    def _tokens(self, x):
        b, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self._pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward_all(self, x):
        x = self._tokens(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_all(x)[:, 0]


class VisionTransformer4K(nn.Module):
    def __init__(
        self,
        num_classes=0,
        img_size=(224,),
        input_embed_dim=384,
        output_embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        **_,
    ):
        super().__init__()
        dim = output_embed_dim
        self.num_features = self.embed_dim = dim
        self.phi = nn.Sequential(
            nn.Linear(input_embed_dim, output_embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
        )
        n_patches = int(img_size[0] // 16) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    inplace_softmax=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_encoding(self, x, w, h):
        n_patch = x.shape[1] - 1
        n_ref = self.pos_embed.shape[1] - 1
        if n_patch == n_ref and w == h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, 0]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w + 0.1, h + 0.1
        side = math.sqrt(n_ref)
        patch_pos = nn.functional.interpolate(
            patch_pos.reshape(1, int(side), int(side), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / side, h0 / side),
            mode="bicubic",
        )
        assert int(w0) == patch_pos.shape[-2] and int(h0) == patch_pos.shape[-1]
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pos.unsqueeze(0), patch_pos), dim=1)

    def _tokens(self, x):
        b, _, w, h = x.shape
        x = self.phi(x.flatten(2, 3).transpose(1, 2))
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x = x + self._pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward_all(self, x):
        x = self._tokens(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x):
        return self.forward_all(x)[:, 0]


def _vit_small():
    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
    )


def _vit4k_xs():
    return VisionTransformer4K(
        patch_size=16,
        input_embed_dim=384,
        output_embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
    )


def _load_weights(model, path):
    state = torch.load(path, map_location="cpu")
    if "teacher" in state:
        state = state["teacher"]
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=False)
    return model


def eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class HIPT4K(nn.Module):
    def __init__(self, model256_path, model4k_path, device="cuda"):
        super().__init__()
        self.device256 = torch.device(device)
        self.device4k = torch.device(device)

        self.model256 = _load_weights(_vit_small(), model256_path)
        self.model4k = _load_weights(_vit4k_xs(), model4k_path)
        for p in self.model256.parameters():
            p.requires_grad = False
        for p in self.model4k.parameters():
            p.requires_grad = False
        self.model256.eval().to(self.device256)
        self.model4k.eval().to(self.device4k)

    @staticmethod
    def _prepare_image(x, patch_size=256):
        _, _, w, h = x.shape
        load_size = (w - w % patch_size, h - h % patch_size)
        x = transforms.CenterCrop(load_size)(x)
        return x, w // patch_size, h // patch_size

    def forward_all256(self, x):
        batch, w256, h256 = self._prepare_image(x)
        batch = batch.unfold(2, 256, 256).unfold(3, 256, 256)
        batch = rearrange(batch, "b c p1 p2 w h -> (b p1 p2) c w h")

        cls_out = []
        sub_out = []
        for start in range(0, batch.shape[0], 256):
            x256 = batch[start:start + 256].to(self.device256, non_blocking=True)
            tokens = self.model256.forward_all(x256).cpu()
            cls_out.append(tokens[:, 0])
            sub_out.append(tokens[:, 1:])

        cls_out = torch.vstack(cls_out)
        sub_out = torch.vstack(sub_out)
        cls_out = (
            cls_out.reshape(w256, h256, 384)
            .transpose(0, 1)
            .transpose(0, 2)
            .unsqueeze(0)
        )
        sub_out = (
            sub_out.reshape(w256, h256, 16, 16, 384)
            .permute(4, 0, 1, 2, 3)
            .unsqueeze(0)
        )
        return cls_out, sub_out

    def forward_all4k(self, cls256):
        _, _, w256, h256 = cls256.shape
        tokens = self.model4k.forward_all(cls256.to(self.device4k, non_blocking=True))
        cls4k = tokens[:, 0]
        sub4k = tokens[:, 1:].reshape(1, w256, h256, 192).permute(0, 3, 1, 2)
        return cls4k, sub4k
