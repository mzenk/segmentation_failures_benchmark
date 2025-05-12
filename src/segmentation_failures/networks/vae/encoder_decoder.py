"""
This code is adapted from David Zimmerer's VAE implementation. Credits to him.
"""

import numpy as np
import torch.nn as nn

from segmentation_failures.networks.vae.utils import ConvBlock, ConvModule, NoOp


# Basic Generator
class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        z_dim=256,
        h_size=(256, 128, 64),
        strides=None,
        kernel_size=3,
        upsample_op=nn.ConvTranspose2d,
        normalization_op=nn.InstanceNorm2d,
        activation_op=nn.LeakyReLU,
        conv_params=None,
        activation_params=None,
        block_op=None,
        block_params=None,
        to_1x1=True,
    ):

        super(Generator, self).__init__()

        if strides is None:
            strides = [2 for _ in range(len(h_size))]

        if conv_params is None:
            conv_params = {}

        n_channels = image_size[0]
        img_size = np.array(image_size[1:])

        if not isinstance(h_size, list) and not isinstance(h_size, tuple):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        elif len(h_size) < 2:
            raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        # We need to know how many layers we will use at the beginning
        img_size_new = img_size // np.prod(strides, axis=0)
        if np.min(img_size_new) < 2 and z_dim is not None:
            raise AttributeError("h_size to long, one image dimension has already perished")

        # --v Start block
        # start_block = []

        # Z_size random numbers

        if not to_1x1:
            kernel_size_start = [min(3, i) for i in img_size_new]
        else:
            kernel_size_start = img_size_new.tolist()

        if z_dim is not None:
            self.start = ConvModule(
                z_dim,
                h_size_bot,
                conv_op=nn.ConvTranspose2d if len(img_size_new) == 2 else nn.ConvTranspose3d,
                # for this operation, the transpose convolution works in any case (stride == 1)
                conv_params=dict(
                    kernel_size=kernel_size_start, stride=1, padding=0, bias=False, **conv_params
                ),
                normalization_op=normalization_op,
                normalization_params={},
                activation_op=activation_op,
                activation_params=activation_params,
            )

        else:
            self.start = NoOp()

        # --v Middle block (Done until we reach ? x image_size/2 x image_size/2)
        self.middle_blocks = nn.ModuleList()

        for idx in range(1, len(h_size)):
            h_size_top = h_size[idx]
            curr_stride = strides[idx - 1]
            # without output padding, the shape may not be correct
            conv_params["output_padding"] = (
                np.array(curr_stride) + 2 - kernel_size
            ).tolist()  # padding == 1 assumed
            if block_op is not None and not isinstance(block_op, str):
                self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=upsample_op,
                    conv_params=dict(
                        kernel_size=kernel_size,
                        stride=curr_stride,
                        padding=1,
                        bias=False,
                        **conv_params,
                    ),
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new * curr_stride

        # --v End block
        # without output padding, the shape may not be correct
        conv_params["output_padding"] = (
            np.array(strides[-1]) + 2 - kernel_size
        ).tolist()  # padding == 1 assumed
        self.end = ConvModule(
            h_size_bot,
            n_channels,
            conv_op=upsample_op,
            conv_params=dict(
                kernel_size=kernel_size, stride=strides[-1], padding=1, bias=False, **conv_params
            ),
            normalization_op=None,
            activation_op=None,
        )

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


# Basic Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        image_size,
        z_dim=256,
        h_size=(64, 128, 256),
        kernel_size=3,
        strides=None,  # results in 2 for each layer
        conv_op=nn.Conv2d,
        normalization_op=nn.InstanceNorm2d,
        activation_op=nn.LeakyReLU,
        conv_params=None,
        activation_params=None,
        block_op=None,
        block_params=None,
        to_1x1=True,
    ):
        super(Encoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        if strides is None:
            strides = [2 for _ in range(len(h_size))]

        n_channels = image_size[0]
        img_size_new = np.array(image_size[1:])

        if not isinstance(h_size, (list, tuple)):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        # elif len(h_size) < 2:
        #     raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        # --v Start block
        self.start = ConvModule(
            n_channels,
            h_size_bot,
            conv_op=conv_op,
            conv_params=dict(
                kernel_size=kernel_size, stride=strides[0], padding=1, bias=False, **conv_params
            ),
            normalization_op=normalization_op,
            normalization_params={},
            activation_op=activation_op,
            activation_params=activation_params,
        )
        img_size_new = img_size_new // strides[0]

        # --v Middle block (Done until we reach ? x 4 x 4)
        self.middle_blocks = nn.ModuleList()

        for layer_idx in range(1, len(h_size)):
            h_size_top = h_size[layer_idx]
            curr_stride = strides[layer_idx]
            if isinstance(curr_stride, int):
                curr_stride = [curr_stride] * len(img_size_new)

            if block_op is not None and not isinstance(block_op, str):
                self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvModule(
                    h_size_bot,
                    h_size_top,
                    conv_op=conv_op,
                    conv_params=dict(
                        kernel_size=kernel_size,
                        stride=curr_stride,
                        padding=1,
                        bias=False,
                        **conv_params,
                    ),
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new // curr_stride

            if np.min(img_size_new) < 2 and z_dim is not None:
                raise ValueError("h_size to long, one image dimension has already perished")

        # --v End block
        if not to_1x1:
            kernel_size_end = [min(3, i) for i in img_size_new]
        else:
            kernel_size_end = img_size_new.tolist()

        if z_dim is not None:
            self.end = ConvModule(
                h_size_bot,
                z_dim,
                conv_op=conv_op,
                conv_params=dict(
                    kernel_size=kernel_size_end, stride=1, padding=0, bias=False, **conv_params
                ),
                normalization_op=None,
                activation_op=None,
            )

            if to_1x1:
                self.output_size = (z_dim, *[1 for _ in img_size_new])
            else:
                self.output_size = (
                    z_dim,
                    *[i - (j - 1) for i, j in zip(img_size_new, kernel_size_end)],
                )
        else:
            self.end = NoOp()
            self.output_size = img_size_new

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


class EncoderLiu(nn.Module):
    def __init__(
        self,
        image_size,
        z_dim=256,
        h_size=(16, 32, 64, 128, 256),
        kernel_size=3,
        strides=None,  # results in 2 for each layer
        conv_op=nn.Conv2d,
        normalization_op=nn.InstanceNorm2d,
        activation_op=nn.LeakyReLU,
        conv_params=None,
        activation_params=None,
        block_op=None,
        block_params=None,
        to_1x1=True,
    ):
        super().__init__()

        if conv_params is None:
            conv_params = {}

        if strides is None:
            strides = [2 for _ in range(len(h_size))]

        n_channels = image_size[0]
        img_size_new = np.array(image_size[1:])

        if not isinstance(h_size, (list, tuple)):
            raise AttributeError("h_size has to be either a list or tuple or an int")

        # --v Start block. Fixed kernel size of 3 and stride of 1
        h_size_bot = 8
        self.start = ConvModule(
            n_channels,
            h_size_bot,
            conv_op=conv_op,
            conv_params=dict(kernel_size=3, stride=1, padding=1, bias=False, **conv_params),
            normalization_op=normalization_op,
            normalization_params={},
            activation_op=activation_op,
            activation_params=activation_params,
        )

        # --v Middle blocks
        self.middle_blocks = nn.ModuleList()

        for layer_idx in range(len(h_size)):
            h_size_top = h_size[layer_idx]
            curr_stride = strides[layer_idx]
            if isinstance(curr_stride, int):
                curr_stride = [curr_stride] * len(img_size_new)

            if block_op is not None and not isinstance(block_op, str):
                self.middle_blocks.append(block_op(h_size_bot, **block_params))

            self.middle_blocks.append(
                ConvBlock(
                    h_size_bot,
                    h_size_top,
                    n_convs=3,
                    initial_stride=curr_stride,
                    conv_op=conv_op,
                    conv_params=dict(
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(0.5 * kernel_size),
                        bias=False,
                        **conv_params,
                    ),
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new // curr_stride

            if np.min(img_size_new) < 2 and z_dim is not None:
                raise ValueError("h_size to long, one image dimension has already perished")

        # --v End block
        if not to_1x1:
            kernel_size_end = [min(3, i) for i in img_size_new]
        else:
            kernel_size_end = img_size_new.tolist()

        if z_dim is None:
            raise ValueError
        self.end = ConvModule(
            h_size_bot,
            z_dim,
            conv_op=conv_op,
            conv_params=dict(
                kernel_size=kernel_size_end, stride=1, padding=0, bias=False, **conv_params
            ),
            normalization_op=None,
            activation_op=None,
        )

        if to_1x1:
            self.output_size = (z_dim, *[1 for _ in img_size_new])
        else:
            self.output_size = (
                z_dim,
                *[i - (j - 1) for i, j in zip(img_size_new, kernel_size_end)],
            )

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output


class GeneratorLiu(nn.Module):
    def __init__(
        self,
        image_size,
        z_dim=256,
        h_size=(256, 128, 64, 32, 16),
        strides=None,
        kernel_size=3,
        upsample_op=nn.ConvTranspose2d,
        normalization_op=nn.InstanceNorm2d,
        activation_op=nn.LeakyReLU,
        conv_params=None,
        activation_params=None,
        block_op=None,
        block_params=None,
        to_1x1=True,
    ):
        super().__init__()

        if strides is None:
            strides = [2 for _ in range(len(h_size))]

        if conv_params is None:
            conv_params = {}

        n_channels = image_size[0]
        img_size = np.array(image_size[1:])

        if not isinstance(h_size, list) and not isinstance(h_size, tuple):
            raise AttributeError("h_size has to be either a list or tuple or an int")
        elif len(h_size) < 2:
            raise AttributeError("h_size has to contain at least three elements")
        else:
            h_size_bot = h_size[0]

        # We need to know how many layers we will use at the beginning
        img_size_new = img_size // np.prod(strides, axis=0)
        if np.min(img_size_new) < 2 and z_dim is not None:
            raise AttributeError("h_size to long, one image dimension has already perished")

        # --v Start block
        # start_block = []

        # Z_size random numbers

        if not to_1x1:
            kernel_size_start = [min(3, i) for i in img_size_new]
        else:
            kernel_size_start = img_size_new.tolist()

        if z_dim is None:
            raise NotImplementedError
        self.start = ConvModule(
            z_dim,
            h_size_bot,
            conv_op=nn.ConvTranspose2d if len(img_size_new) == 2 else nn.ConvTranspose3d,
            # for this operation, the transpose convolution works in any case (stride == 1)
            conv_params=dict(
                kernel_size=kernel_size_start, stride=1, padding=0, bias=False, **conv_params
            ),
            normalization_op=normalization_op,
            normalization_params={},
            activation_op=activation_op,
            activation_params=activation_params,
        )

        # --v Middle block (Done until we reach ? x image_size/2 x image_size/2)
        self.middle_blocks = nn.ModuleList()

        for idx in range(1, len(h_size)):
            h_size_top = h_size[idx]
            curr_stride = strides[idx - 1]
            self.middle_blocks.append(
                ConvBlock(
                    h_size_bot,
                    h_size_top,
                    n_convs=3,
                    initial_stride=curr_stride,
                    conv_op=upsample_op,
                    conv_params=dict(
                        kernel_size=kernel_size,
                        stride=1,
                        padding=int(0.5 * kernel_size),
                        bias=False,
                        **conv_params,
                    ),
                    normalization_op=normalization_op,
                    normalization_params={},
                    activation_op=activation_op,
                    activation_params=activation_params,
                )
            )

            h_size_bot = h_size_top
            img_size_new = img_size_new * curr_stride

        # --v End block
        self.end = nn.Sequential(
            ConvBlock(
                h_size_bot,
                8,
                n_convs=0,  # results in only upsample transpose conv.
                initial_stride=strides[-1],
                conv_op=upsample_op,
                conv_params={},
                normalization_op=normalization_op,
                normalization_params={},
                activation_op=activation_op,
                activation_params=activation_params,
            ),
            ConvModule(
                8,
                n_channels,
                conv_op=upsample_op,
                conv_params=dict(
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                normalization_op=None,
                activation_op=None,
            ),
        )

    def forward(self, inpt, **kwargs):
        output = self.start(inpt, **kwargs)
        for middle in self.middle_blocks:
            output = middle(output, **kwargs)
        output = self.end(output, **kwargs)
        return output
