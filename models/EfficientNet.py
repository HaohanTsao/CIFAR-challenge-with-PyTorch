# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from functools import partial

# %%
# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "width_coefficient",
        "depth_coefficient",
        "image_size",
        "dropout_rate",
        "num_classes",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "drop_connect_rate",
        "depth_divisor",
        "min_depth",
        "include_top",
    ],
)

# Parameters for an individual model block
BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "num_repeat",
        "kernel_size",
        "stride",
        "expand_ratio",
        "input_filters",
        "output_filters",
        "se_ratio",
        "id_skip",
    ],
)

# %%


def swish(x):
    return x * torch.sigmoid(x)


def drop_connect(inputs, rate, training):
    """隨機drop掉一些網絡，防止overfitting"""
    assert 0 <= rate <= 1, "p must be in range of [0,1]"

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - rate

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """
    在efficientNet中的論文使用Mobile Inverted Residual Bottleneck Block作為basic block
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_momentum = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        input_channels = self._block_args.input_filters  # number of input channels
        output_channels = (
            self._block_args.input_filters * self._block_args.expand_ratio
        )  # number of output channels
        """
        blocks_args = [
            BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=32, output_filters=16, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=2, kernel_size=3, stride=[2], expand_ratio=6, input_filters=16, output_filters=24, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=24, output_filters=40, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=40, output_filters=80, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=80, output_filters=112, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=112, output_filters=192, se_ratio=0.25, id_skip=True),
            BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=192, output_filters=320, se_ratio=0.25, id_skip=True),]

        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'EfficientNetB0': (1.0, 1.0, 224, 0.2),
            'EfficientNetB1': (1.0, 1.1, 240, 0.2),
            'EfficientNetB2': (1.1, 1.2, 260, 0.3),
            'EfficientNetB3': (1.2, 1.4, 300, 0.3),
            'EfficientNetB4': (1.4, 1.8, 380, 0.4),
            'EfficientNetB5': (1.6, 2.2, 456, 0.4),
            'EfficientNetB6': (1.8, 2.6, 528, 0.5),
            'EfficientNetB7': (2.0, 3.1, 600, 0.5),
            'EfficientNetB8': (2.2, 3.6, 672, 0.5),
            'EfficientNetL2': (4.3, 5.3, 800, 0.5),
        }
        """
        # Expansion
        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                bias=False,
                padding=0,
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=output_channels,
                momentum=self._bn_momentum,
                eps=self._bn_eps,
            )

        # Depthwise
        kernel_size = self._block_args.kernel_size
        stride = self._block_args.stride
        self._depthwise_conv = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            groups=output_channels,  # groups makes it depthwise
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=(1 if kernel_size == 3 else 2),
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=output_channels, momentum=self._bn_momentum, eps=self._bn_eps
        )

        # Squeeze and Excitation
        if self.has_se:
            squeezed_channels = max(
                1, int(input_channels * self._block_args.se_ratio)
            )  # 0.25 basically
            self._se_reduce = nn.Conv2d(
                in_channels=output_channels,
                out_channels=squeezed_channels,
                kernel_size=1,
            )
            self._se_expand = nn.Conv2d(
                in_channels=squeezed_channels,
                out_channels=output_channels,
                kernel_size=1,
            )

        # Pointwise
        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(
            in_channels=output_channels,
            out_channels=final_oup,
            kernel_size=1,
            bias=False,
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_momentum, eps=self._bn_eps
        )

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = swish(x)

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, rate=drop_connect_rate, training=self.training)
            x = x + inputs  # 殘差連接
        return x


# %%
def round_filters(filters, global_params):
    """
    Calculate and round number of filters based on width multiplier.
    Use width_coefficient, depth_divisor and min_depth of global_params.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters

    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """
    Calculate module's repeat number of a block based on depth multiplier.
    Use depth_coefficient of global_params.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list)  # blocks_args should be a list
        assert len(blocks_args) > 0  # block args must be greater than 0
        self._global_params = global_params
        self._blocks_args = blocks_args

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        image_size = global_params.image_size
        in_channels = 3  # rgb

        # stem
        out_channels = round_filters(
            32, self._global_params
        )  # number of output channels
        self._conv_stem = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Build blocks
        self._blocks = nn.ModuleList([])
        count_blocks = 1
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(block_args, self._global_params, image_size=image_size)
            )
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params, image_size=image_size)
                )
            count_blocks += 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def forward(self, inputs):
        # Convolution layers
        # Stem
        x = swish(self._bn0(self._conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = swish(self._bn1(self._conv_head(x)))
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x


# %%
def efficientnet_params(model_name):
    """
    Map EfficientNet model name to parameter coefficients.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        "EfficientNetB0": (1.0, 1.0, 224, 0.2),
        "EfficientNetB1": (1.0, 1.1, 240, 0.2),
        "EfficientNetB2": (1.1, 1.2, 260, 0.3),
        "EfficientNetB3": (1.2, 1.4, 300, 0.3),
        "EfficientNetB4": (1.4, 1.8, 380, 0.4),
        "EfficientNetB5": (1.6, 2.2, 456, 0.4),
        "EfficientNetB6": (1.8, 2.6, 528, 0.5),
        "EfficientNetB7": (2.0, 3.1, 600, 0.5),
        "EfficientNetB8": (2.2, 3.6, 672, 0.5),
        "EfficientNetL2": (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet(model_name, num_classes=100, drop_connect_rate=0.2, include_top=True):
    """
    Create BlockArgs and GlobalParams for efficientnet model.
    """

    # Blocks args for the whole model(EfficientNetB0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_args = [
        BlockArgs(
            num_repeat=1,
            kernel_size=3,
            stride=[1],
            expand_ratio=1,
            input_filters=32,
            output_filters=16,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=2,
            kernel_size=3,
            stride=[2],
            expand_ratio=6,
            input_filters=16,
            output_filters=24,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=2,
            kernel_size=5,
            stride=[2],
            expand_ratio=6,
            input_filters=24,
            output_filters=40,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=3,
            kernel_size=3,
            stride=[2],
            expand_ratio=6,
            input_filters=40,
            output_filters=80,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=3,
            kernel_size=5,
            stride=[1],
            expand_ratio=6,
            input_filters=80,
            output_filters=112,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=4,
            kernel_size=5,
            stride=[2],
            expand_ratio=6,
            input_filters=112,
            output_filters=192,
            se_ratio=0.25,
            id_skip=True,
        ),
        BlockArgs(
            num_repeat=1,
            kernel_size=3,
            stride=[1],
            expand_ratio=6,
            input_filters=192,
            output_filters=320,
            se_ratio=0.25,
            id_skip=True,
        ),
    ]

    net_params = efficientnet_params(model_name)

    global_params = GlobalParams(
        width_coefficient=net_params[0],
        depth_coefficient=net_params[1],
        image_size=net_params[2],
        dropout_rate=net_params[3],
        num_classes=num_classes,
        batch_norm_momentum=0.9,  # 原來是0.99
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return EfficientNet(blocks_args, global_params)


# %%
# test
def test(model_fn):
    model = model_fn
    try:
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print("Your model is ready!")
    except Exception as e:
        error_message = (
            "There are problems with the model. Please check the model's architecture."
        )
        raise Exception(error_message) from e


# %%
# model_fn = efficientnet('EfficientNetB7', num_classes=100)
# test(model_fn)
# %%
