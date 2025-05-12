from copy import copy
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
from loguru import logger
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
)
from monai.networks.nets.dynunet import DynUNet, DynUNetSkipLayer
from torch import nn


def get_kernels_strides(patch_size: list[int], spacings: list[float]):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    current_size = copy(patch_size)
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, current_size)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(current_size, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {patch_size[idx]} in the spatial dimension {idx}."
                )
        current_size = [i / j for i, j in zip(current_size, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


# network
def get_network(
    out_channels,
    spatial_dims,
    in_channels,
    patch_size,
    spacings,
    dropout=0,
    num_dropout_units=0,
    deep_supervision=False,
    checkpoint_path=None,
    **dynunet_kwargs,
):
    if spacings is None or patch_size is None:
        raise ValueError(
            "spacings and patch_size must be provided to determine kernels and strides"
        )
    kernels, strides = get_kernels_strides(patch_size, spacings)
    logger.info("DynUNet determined the following kernels/strides given patch size and spacings:")
    logger.info("Kernels" + str(kernels))
    logger.info("Strides" + str(strides))
    net = EnhancedDynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=deep_supervision,
        dropout=dropout,
        num_dropout_units=num_dropout_units,
        **dynunet_kwargs,
    )
    # net = DynUNet(
    #     spatial_dims=spatial_dims,
    #     in_channels=in_channels,
    #     out_channels=out_channels,
    #     kernel_size=kernels,
    #     strides=strides,
    #     upsample_kernel_size=strides[1:],
    #     norm_name="instance",
    #     deep_supervision=deep_supr_num > 0,
    #     deep_supr_num=deep_supr_num,
    #     dropout=dropout,
    #     **dynunet_kwargs,
    # )

    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            net.load_state_dict(torch.load(checkpoint_path))
            print("pretrained checkpoint: {} loaded".format(checkpoint_path))
        else:
            print("no pretrained checkpoint")
    return net


class EnhancedDynUNet(DynUNet):
    """Modification of Dynunet for advanced dropout strategies.

    This class adds dropout units symmetrically around the bottleneck, i.e. for
    `num_dropout_units == 1` only the bottleneck, for 3 bottleneck plus 1 on each side (encoder/decoder).
    If the number is even, the class will only add the next lower odd number of dropout units.
    The output block never has dropout.
    Args:
        num_dropout_units (int, optional): Number of blocks that should have dropout enabled. Defaults to 0.
        append_dropout (bool, optional): If True, the dropout layer is appended to the block. Defaults to True.
        blocks_per_stage (int, optional): Number of blocks per stage. Defaults to 1.
        filter_scaling (float, optional): Scaling factor for the number of filters per conv. Defaults to 1.0.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        num_dropout_units=0,
        append_dropout=True,
        blocks_per_stage: int = 1,
        filter_scaling: float = 1.0,
    ):
        """Copied and slightly adapted from monai.networks.nets.dynunet.DynUNet."""
        super(DynUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        # --- CHANGED/ADDED LINES below
        self.conv_block = UnetStackedResBlock if res_block > 0 else UnetStackedBlock
        if num_dropout_units > 0 and num_dropout_units % 2 == 0:
            logger.warning(
                f"An even number of dropout ({num_dropout_units}) units was specified. "
                f"This class only supports odd numbers. Using {num_dropout_units - 1} instead."
            )
            num_dropout_units -= 1
        self.num_dropout_units = num_dropout_units
        if self.num_dropout_units == 0 and self.dropout > 0:
            logger.warning(
                "Dropout rate is > 0 but num_dropout_units == 0. Will continue without dropout."
            )
        self.append_dropout = append_dropout
        self.blocks_per_stage = blocks_per_stage
        # ---
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [
                min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))
            ]
            # ADDED (scaling)
            self.filters = [int(f * filter_scaling) for f in self.filters]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.deep_supervision = deep_supervision
        self.deep_supr_num = len(self.upsamples) - 1 if deep_supervision else 0
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

        def create_skips(index, downsamples, upsamples, bottleneck, superheads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """

            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            if superheads is None:
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)
                return DynUNetSkipLayer(
                    index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer
                )

            super_head_flag = False
            if index == 0:  # don't associate a supervision head with self.input_block
                rest_heads = superheads
            else:
                if len(superheads) > 0:
                    super_head_flag = True
                    rest_heads = superheads[1:]
                else:
                    rest_heads = nn.ModuleList()

            # create the next layer down, this will stop at the bottleneck layer
            next_layer = create_skips(
                1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads
            )
            if super_head_flag:
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=self.heads,
                    super_head=superheads[0],
                )

            return DynUNetSkipLayer(
                index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer
            )

        if not self.deep_supervision:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
            )
        else:
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1],
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

    def get_input_block(self):
        dropout = 0
        if self.num_dropout_units >= 2 * len(self.filters) - 1:
            # composition of self.filters:
            # 0    -- input block
            # 1:-1 -- downsamples
            # -1   -- bottleneck
            # -1:0 -- upsamples
            # -> upsample filters are re-used hence len(filters) + len(filters) - 1
            dropout = self.dropout
        layer = self.conv_block(
            1,
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=dropout * (1 - self.append_dropout),
        )
        if self.append_dropout and dropout > 0:
            return nn.Sequential(layer, nn.Dropout(self.dropout))
        return layer

    def get_output_block(self, idx: int):
        # never use dropout here, this is the last layer before the softmax!
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=0)

    def get_bottleneck(self):
        dropout = 0
        if self.num_dropout_units > 0:
            dropout = self.dropout
        bottleneck = self.conv_block(
            self.blocks_per_stage,
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=dropout * (1 - self.append_dropout),
        )
        if self.append_dropout and dropout > 0:
            return nn.Sequential(bottleneck, nn.Dropout(self.dropout))
        return bottleneck

    def get_module_list(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: Type[nn.Module],
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ):
        layers = []
        dropout = [0.0] * len(in_channels)
        for idx, _ in enumerate(dropout):
            if idx < 0.5 * (self.num_dropout_units - 1):
                dropout[idx] = self.dropout
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel, drp in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size, dropout
            ):
                # Assume upsampling branch
                params = {
                    # no n_blocks because we use the default decoder
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": drp * (1 - self.append_dropout),
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                if self.append_dropout and drp > 0:
                    layer = FlexibleSequential(layer, nn.Dropout(drp))
                    # The problem with nn.Sequential is that it assumes one input arg.
                    # However, the up-layer in dynunet accepts two arguments (skip input and input from lower layer)
                layers.append(layer)
        else:
            # Assume downsampling branch
            dropout = dropout[::-1]
            for in_c, out_c, kernel, stride, drp in zip(
                in_channels, out_channels, kernel_size, strides, dropout
            ):
                params = {
                    "n_blocks": self.blocks_per_stage,
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": drp * (1 - self.append_dropout),
                }
                layer = conv_block(**params)
                if self.append_dropout and drp > 0:
                    layer = nn.Sequential(layer, nn.Dropout(drp))
                layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.skip_layers(x)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads:
                out_all.append(feature_map)
            return out_all
        return out


class FlexibleSequential(nn.Sequential):
    # This just allows to pass any number of arguments to forward
    def forward(self, *inputs):
        for idx, module in enumerate(self):
            if idx == 0:
                input = module(*inputs)
            else:
                input = module(input)
        return input


class UnetStackedBlock(nn.Module):
    """
    Similar to monai.networks.blocks.dynunet_blocks.UnetResBlock but with multiple residual blocks

    Args:
        n_blocks: number of residual blocks.
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.

    """

    def __init__(
        self,
        n_blocks: int,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        conv_block=UnetBasicBlock,
    ):
        super().__init__()
        assert n_blocks > 0
        blocks = [
            conv_block(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                    act_name=act_name,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class UnetStackedResBlock(UnetStackedBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, conv_block=UnetResBlock)
