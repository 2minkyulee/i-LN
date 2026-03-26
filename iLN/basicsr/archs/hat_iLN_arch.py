import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs import hat_arch as hat_base


glob_eps = None

# The proposed iLN.
# 1) Spatially holistic norm and
# 2) Denorm (=input adaptive rescaling)
class iLN(nn.Module):

    def __init__(self, dim, *args, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.last_var = None

    def forward(self, x):
        mean = x.mean(dim=list(range(1, x.dim())), keepdim=True)
        var = x.var(dim=list(range(1, x.dim())), keepdim=True, unbiased=False)
        self.last_var = var
        x = (x - mean) / torch.sqrt(var + glob_eps)
        return x * self.gamma + self.beta

    def denorm(self, x):
        return x * torch.sqrt(self.last_var + glob_eps)

# Used in cases where denorm is followed right after norm.
# (Ignoring mean shifting,) norm+affine+denorm simply becomes affine, which is more efficient and stable.
class Affine(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta



# Everything is identical to the original HAB block, except for "denorm".
class HAB(hat_base.HAB):

    def merge_attention_output(self, shortcut, attn_x, conv_x):
        return shortcut + self.norm1.denorm(self.drop_path(attn_x) + conv_x * self.conv_scale)

    def merge_mlp_output(self, x):
        return x + self.norm2.denorm(self.drop_path(self.mlp(self.norm2(x))))

# Everything is identical to the original OCAB block, except for "denorm".
class OCAB(hat_base.OCAB):

    def merge_projection_output(self, shortcut, x):
        return self.norm1.denorm(self.proj(x)) + shortcut

    def merge_mlp_output(self, x):
        return x + self.norm2.denorm(self.mlp(self.norm2(x)))




# Modified HAT with iLN.
# 1) We change LayerNorm to iLN in all the places (HAB, OCAB, patch embedding/unembedding, and final norm).
# 2) Removing the "final" LayerNorm (which does not have any residual connection), and substituting it with Affine is important.
# 3) We additionally do std_norm

@ARCH_REGISTRY.register()
class HAT_iLN(hat_base.HAT):
    hab_cls = HAB
    ocab_cls = OCAB
    patch_embed_norm_cls = Affine
    patch_unembed_norm_cls = Affine
    final_norm_cls = Affine  # <---- removing the vanilla LN here helps a lot.

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=iLN,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 eps=1e-4,
                 **kwargs):
        global glob_eps
        glob_eps = eps
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            **kwargs)


# alias to prevent typos for users
ARCH_REGISTRY._do_register('HAT-iLN', HAT_iLN)
