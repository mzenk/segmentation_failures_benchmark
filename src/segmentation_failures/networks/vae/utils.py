"""
This code is adapted from David Zimmerer's VAE implementation. Credits to him.
"""

import warnings

import torch.nn as nn
import torch.nn.functional as F


class NoOp(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoOp, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x


def non_lin(
    nn_module,
    normalization="",
    activation="LReLU",
    feat_size=None,
):
    """Non Lienar activation unit"""

    if normalization == "batch":
        assert feat_size is not None
        nn_module.append(nn.BatchNorm2d(feat_size))
    elif normalization == "weight":
        nn_module[-1] = nn.utils.weight_norm(nn_module[-1])
    elif normalization == "instance":
        assert feat_size is not None
        nn_module.append(nn.InstanceNorm2d(feat_size))

    if activation == "LReLU":
        nn_module.append(nn.LeakyReLU(0.2))
    elif activation == "ELU":
        nn_module.append(nn.ELU())
    elif activation == "ReLU":
        nn_module.append(nn.ReLU())
    elif activation == "SELU":
        nn_module.append(nn.SELU())
    else:
        warnings.warn(
            "Will not use any non linear activation function", RuntimeWarning, stacklevel=2
        )


class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super(ConvDownsample, self).__init__()

        self.conv_op = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False
        )
        self.down_op = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.down_op(self.conv_op(x))


class ConvUpsample(nn.Module):
    # adapted from https://medium.com/miccai-educational-initiative/tutorial-abdominal-ct-image-synthesis-with-variational-autoencoders-using-pytorch-933c29bb1c90  # noqa: B950
    # supposedly reduces checkerboard artifacts
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
        conv_op=nn.Conv2d,
        **kwargs,
    ):
        # NOTE: stride is assumed to be the desired upsampling factor. The convolution performed here will have stride 1.
        super(ConvUpsample, self).__init__()
        self.scale_factor = stride
        assert isinstance(kernel_size, int)
        if len(kwargs) > 0:
            warnings.warn(f"ConvUpsample: Unused kwargs: {kwargs}", stacklevel=2)
        self.conv = conv_op(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


Conv2DUpsample = ConvUpsample


class Conv3DUpsample(ConvUpsample):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=False,
        conv_op=nn.Conv3d,
        **kwargs,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias, conv_op, **kwargs
        )


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=nn.LeakyReLU,
        activation_params=None,
    ):
        super(ConvModule, self).__init__()

        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}
        self.activation_params = activation_params
        if self.activation_params is None:
            self.activation_params = {}
        self.normalization_params = normalization_params
        if self.normalization_params is None:
            self.normalization_params = {}

        self.conv = None
        if conv_op is not None and not isinstance(conv_op, str):
            self.conv = conv_op(in_channels, out_channels, **self.conv_params)

        self.normalization = None
        if normalization_op is not None and not isinstance(normalization_op, str):
            self.normalization = normalization_op(out_channels, **self.normalization_params)

        self.activation = None
        if activation_op is not None and not isinstance(activation_op, str):
            self.activation = activation_op(**self.activation_params)

    def forward(
        self, input, conv_add_input=None, normalization_add_input=None, activation_add_input=None
    ):
        x = input

        if self.conv is not None:
            if conv_add_input is None:
                x = self.conv(x)
            else:
                x = self.conv(x, **conv_add_input)

        if self.normalization is not None:
            if normalization_add_input is None:
                x = self.normalization(x)
            else:
                x = self.normalization(x, **normalization_add_input)

        if self.activation is not None:
            if activation_add_input is None:
                x = self.activation(x)
            else:
                x = self.activation(x, **activation_add_input)

        # nn.functional.dropout(x, p=0.95, training=True)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_featmaps,
        out_featmaps,
        n_convs,
        initial_stride,
        conv_op=nn.Conv2d,
        conv_params=None,
        normalization_op=nn.InstanceNorm2d,
        normalization_params=None,
        activation_op=nn.ReLU,
        activation_params=None,
    ):
        super(ConvBlock, self).__init__()

        self.n_featmaps = in_featmaps
        self.n_convs = n_convs
        self.conv_params = conv_params
        if self.conv_params is None:
            self.conv_params = {}

        initial_kernel_size = 2
        if isinstance(initial_stride, (tuple, list)):
            initial_kernel_size = [2 if s > 1 else 1 for s in initial_stride]
        conv_list = [
            ConvModule(
                in_featmaps,
                out_featmaps,
                conv_op=conv_op,
                conv_params=dict(
                    kernel_size=initial_kernel_size, stride=initial_stride, padding=0, bias=False
                ),
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )
        ]
        for _ in range(self.n_convs):
            conv_layer = ConvModule(
                out_featmaps,
                out_featmaps,
                conv_op=conv_op,
                conv_params=conv_params,
                normalization_op=normalization_op,
                normalization_params=normalization_params,
                activation_op=activation_op,
                activation_params=activation_params,
            )
            conv_list.append(conv_layer)
        self.conv_list = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv_list(x)
