import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.cuda.amp import autocast
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3, channels=None, gamma=2, b=1):
        super().__init__()
        if channels is not None:
            t = int(abs((math.log(channels, 2) + b) / gamma))
            kernel_size = t if t % 2 else t + 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)

class MultiScaleConvBlock(nn.Module):
    def __init__(
            self,
            conv_op: Type[_ConvNd],
            input_channels: int,
            output_channels: int,
            kernel_sizes: List[int] = None,
            stride: int = 1,
            norm_op: Type[nn.Module] = None,
            norm_op_kwargs: dict = None,
            nonlin: Type[nn.Module] = None,
            nonlin_kwargs: dict = None,
            padding: int = None
    ):
        super().__init__()
        self.nonlin_kwargs = nonlin_kwargs.copy() if nonlin_kwargs is not None else {}
        self.nonlin_kwargs['inplace'] = False

        self.project_in = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.project_in = nn.Sequential(
                conv_op(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
                nonlin(**self.nonlin_kwargs) if nonlin else nn.Identity()
            )

        self.split_c = output_channels // 4
        self.last_c = output_channels - self.split_c * 3
        self.dw1 = conv_op(self.last_c, self.last_c, kernel_size=1, bias=False)
        self.dw3 = conv_op(self.split_c, self.split_c, kernel_size=3, padding=1, groups=self.split_c, bias=False)
        self.dw5 = conv_op(self.split_c, self.split_c, kernel_size=5, padding=2, groups=self.split_c, bias=False)
        self.dw7 = conv_op(self.split_c, self.split_c, kernel_size=7, padding=3, groups=self.split_c, bias=False)
        self.norm_act = nn.Sequential(
            norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
            nonlin(**self.nonlin_kwargs) if nonlin else nn.Identity()
        )

        self.fusion_conv = nn.Sequential(
            conv_op(output_channels, output_channels, kernel_size=1, bias=False),
            norm_op(output_channels, **norm_op_kwargs) if norm_op else nn.Identity(),
            nonlin(**self.nonlin_kwargs) if nonlin else nn.Identity()
        )

        self.eca = ECAAttention(channels=output_channels)

    def forward(self, x):

        x = self.project_in(x)
        residual = x

        B, C, H, W = x.shape
        g = 4
        if C % g == 0:
            x = x.view(B, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(B, C, H, W)

        x_split = torch.split(x, [self.split_c, self.split_c, self.split_c, self.last_c], dim=1)

        x3 = self.dw3(x_split[0])
        x5 = self.dw5(x_split[1])
        x7 = self.dw7(x_split[2])
        x1 = self.dw1(x_split[3])

        x_out = torch.cat([x3, x5, x7, x1], dim=1)
        x_out = self.norm_act(x_out)

        x_out = self.fusion_conv(x_out)
        x_out = self.eca(x_out)

        return x_out + residual

class LightweightMambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=2, expand=1, channel_token=False):
        super().__init__()
        self.dim = dim
        self.channel_token = channel_token
        self.norm = nn.LayerNorm(dim) if not channel_token else nn.Identity()

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=False
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            if torch.cuda.is_bf16_supported():
                x = x.type(torch.bfloat16)
            else:
                x = x.type(torch.float32)

        residual = x

        if self.channel_token:
            B, C, H, W = x.shape
            x_flat = x.flatten(2)
            x_norm = self.norm(x_flat)
            x_mamba = self.mamba(x_norm)
            out = x_mamba.reshape(B, C, H, W)
        else:
            B, C, H, W = x.shape
            n_tokens = H * W

            x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
            x_norm = self.norm(x_flat)

            out_fwd = self.mamba(x_norm)

            x_rev = torch.flip(x_norm, dims=[1])
            out_rev = self.mamba(x_rev)
            out_rev = torch.flip(out_rev, dims=[1])

            x_mamba = (out_fwd + out_rev) / 2.0

            out = x_mamba.transpose(-1, -2).reshape(B, C, H, W)

        return out + residual

class ChannelAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y)))).view(b, c, 1, 1)
        return x * y

class DualChannelAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, self.out_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y)

        w_skip, w_mamba = torch.split(y, self.in_channels, dim=1)

        w_skip = w_skip.reshape(b, c, 1, 1)
        w_mamba = w_mamba.reshape(b, c, 1, 1)

        return w_skip, w_mamba

class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, C, H, W = x.shape
        assert C == self.dim
        n_tokens = H * W

        x_flat_h = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm_h = self.norm(x_flat_h)
        out_h = self.mamba(x_norm_h)
        out_h = out_h.transpose(-1, -2).reshape(B, C, H, W)

        x_v = x.transpose(-1, -2)
        x_flat_v = x_v.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm_v = self.norm(x_flat_v)
        out_v = self.mamba(x_norm_v)
        out_v = out_v.transpose(-1, -2).reshape(B, C, W, H).transpose(-1, -2)

        out = (out_h + out_v) / 2.0

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    def forward(self, x):
        if x.dtype == torch.float16:
            if torch.cuda.is_bf16_supported():
                x = x.type(torch.bfloat16)
            else:
                x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out

class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = MultiScaleConvBlock(
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=output_channels,
            stride=stride,
            padding=padding,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs
        )

        self.conv2 = MultiScaleConvBlock(
            conv_op=conv_op,
            input_channels=output_channels,
            output_channels=output_channels,
            stride=1,
            padding=padding,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs
        )

        self.norm1 = nn.Identity()
        self.act1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.act2 = nonlin(**nonlin_kwargs)
        self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride) if use_1x1conv else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int): kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int): features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int): n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int): strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ),
            *[BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=kernel_sizes[0],
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        input_channels = stem_channels

        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                ),
                *[BasicBlockD(
                    conv_op=conv_op,
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        self.attn_modules = nn.ModuleList()
        for s in range(n_stages):
            self.attn_modules.append(DualChannelAttention2D(features_per_stage[s]))

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            feat = self.stages[s](x)
            skip_feat = feat
            mamba_out = self.mamba_layers[s](feat)
            sum_feat = skip_feat + mamba_out
            w_skip, w_mamba = self.attn_modules[s](sum_feat)
            skip_weighted = skip_feat * w_skip
            mamba_weighted = mamba_out * w_mamba
            x = skip_weighted + mamba_weighted
            ret.append(x)
        return ret if self.return_skips else ret[-1]

class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []
        self.light_mamba_layers = nn.ModuleList()

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            decode_stage = nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip if s < n_stages_encoder - 1 else input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                ) for _ in range(n_conv_per_stage[s - 1] - 1)],
                LightweightMambaLayer(
                    dim=input_features_skip,
                    channel_token=False
                )
            )
            stages.append(decode_stage)
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        return seg_outputs[0] if not self.deep_supervision else seg_outputs

class UMambaEnc(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

def get_umamba_enc_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)
    network_class = UMambaEnc

    kwargs = {
        'UMambaEnc': {
            'input_size': configuration_manager.patch_size,
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision, **conv_or_blocks_per_stage,
        **kwargs['UMambaEnc']
    )
    model.apply(InitWeights_He(1e-2))
    return model