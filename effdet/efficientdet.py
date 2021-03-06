""" PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple
from functools import partial

from timm import create_model
from timm.models.layers import create_conv2d, create_pool2d, Swish, get_act_layer
from .config import get_fpn_config, set_config_writeable, set_config_readonly

from absl import flags

FLAGS = flags.FLAGS


_DEBUG = False

_ACT_LAYER = Swish


class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=True)


class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self, in_channels, out_channels, reduction_ratio=1., pad_type='', downsample=None, upsample=None,
            norm_layer=nn.BatchNorm2d, apply_bn=False, conv_after_downsample=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias, act_layer=None)

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            if downsample in ('max', 'avg'):
                stride_size = int(reduction_ratio)
                downsample = create_pool2d(
                     downsample, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type)
            else:
                downsample = Interpolate2d(scale_factor=1./reduction_ratio, mode=downsample)
            self.add_module('downsample', downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))

    # def forward(self, x):
    #     #  here for debugging only
    #     assert x.shape[1] == self.in_channels
    #     if self.reduction_ratio > 1:
    #         if hasattr(self, 'conv') and not self.conv_after_downsample:
    #             x = self.conv(x)
    #         x = self.downsample(x)
    #         if hasattr(self, 'conv') and self.conv_after_downsample:
    #             x = self.conv(x)
    #     else:
    #         if hasattr(self, 'conv'):
    #             x = self.conv(x)
    #         if self.reduction_ratio < 1:
    #             x = self.upsample(x)
    #     return x


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 conv_after_downsample=False, redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample, redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER,
                 apply_resample_bn=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True, redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            reduction = fnode_cfg['reduction']
            combine = FpnCombine(
                feature_info, fpn_config, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction, pad_type=pad_type, downsample=downsample, upsample=upsample,
                norm_layer=norm_layer, apply_resample_bn=apply_resample_bn, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=False, norm_layer=norm_layer, act_layer=act_layer)
            if not conv_bn_relu_pattern:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    apply_bn=config.apply_resample_bn,
                    conv_after_downsample=config.conv_after_downsample,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                conv_after_downsample=config.conv_after_downsample,
                conv_bn_relu_pattern=config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


class HeadNet(nn.Module):

    def __init__(self, config, num_outputs, num_channels_flag=None):
        super(HeadNet, self).__init__()
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, 'head_bn_level_first', False)
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_type = config.head_act_type if getattr(config, 'head_act_type', None) else config.act_type
        act_layer = get_act_layer(act_type) or _ACT_LAYER

        num_channels = num_channels_flag or config.fpn_channels

        # Build convolution repeats
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        if not num_channels_flag is None:
            in_conv = dict(
                in_channels=config.fpn_channels, out_channels=num_channels, kernel_size=3,
                padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
            conv_kwargs = dict(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3,
                padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
            self.conv_rep = nn.ModuleList([conv_fn(**in_conv)]+[conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats-1)])
        else:
            conv_kwargs = dict(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3,
                padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
            self.conv_rep = nn.ModuleList([conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats)])

        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        # This can be organized with repeats first or feature levels first in module lists, the original models
        # and weights were setup with repeats first, levels first is required for efficient torchscript usage.
        self.bn_rep = nn.ModuleList()
        for _ in range(config.box_class_repeats):
            self.bn_rep.append(nn.ModuleList([
                nn.Sequential(OrderedDict([('bn', norm_layer(num_channels))]))
                for _ in range(self.num_levels)]))

        self.act = act_layer(inplace=True)

        # Prediction (output) layer. Has bias with special init reqs, see init fn.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(
            in_channels=num_channels, out_channels=num_outputs * num_anchors, kernel_size=3,
            padding=config.pad_type, bias=True, norm_layer=None, act_layer=None)
        self.predict = conv_fn(**predict_kwargs)

    @torch.jit.ignore()
    def toggle_bn_level_first(self):
        """ Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        with torch.no_grad():
            new_bn_rep = nn.ModuleList()
            for i in range(len(self.bn_rep[0])):
                bn_first = nn.ModuleList()
                for r in self.bn_rep.children():
                    m = r[i]
                    # NOTE original rep first model def has extra Sequential container with 'bn', this was
                    # flattened in the level first definition.
                    bn_first.append(m[0] if isinstance(m, nn.Sequential) else nn.Sequential(OrderedDict([('bn', m)])))
                new_bn_rep.append(bn_first)
            self.bn_level_first = not self.bn_level_first
            self.bn_rep = new_bn_rep

    @torch.jit.ignore()
    def _forward(self, x: List[torch.Tensor], ret_activs=False, level_offset=0) -> List[torch.Tensor]:
        outputs = []
        if ret_activs: activs = []
        for level in range(level_offset,self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)  # this is not allowed in torchscript
                x_level = self.act(x_level)
            if ret_activs: activs.append(x_level)
            outputs.append(self.predict(x_level))
        if ret_activs:
            return activs, outputs
        else:
            return outputs

    def _forward_level_first(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(self.bn_rep):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def forward(self, x: List[torch.Tensor], ret_activs=False, level_offset=0) -> List[torch.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x, ret_activs=ret_activs, level_offset=level_offset)


def _init_weight(m, n='', ):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n or 'anchor_out' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(m, n='', ):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(backbone.feature_info())]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
    return feature_info


class MetaHead(nn.Module):

    def __init__(self,config,pretrain_init=None,num_channels_flag=None):
        super(MetaHead, self).__init__()
        self.num_layers = config.box_class_repeats
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        
        in_channels = config.fpn_channels
        self.num_channels = num_channels = num_channels_flag or config.fpn_channels
        self.num_anchors = num_anchors = len(config.aspect_ratios) * config.num_scales

        '''self.conv_dw_rep = nn.ParameterList([nn.Parameter(torch.randn((in_channels,in_channels,3,3))*((1/in_channels)**0.5))] +
            [nn.Parameter(torch.randn((num_channels,num_channels,3,3))*((1/num_channels)**0.5)) for _ in range(FLAGS.num_conv-1)])
        self.conv_pw_rep = nn.ParameterList([nn.Parameter(torch.randn((num_channels,in_channels,1,1))*((1/in_channels)**0.5))] +
            [nn.Parameter(torch.randn((num_channels,num_channels,1,1))*((1/num_channels)**0.5)) for _ in range(FLAGS.num_conv-1)])'''

        
        #nn.ParameterList([nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_dw.weight'.format(l)]) 
        #    for l in range(self.num_layers)])
        #self.conv_pw_rep = nn.ParameterList([nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_pw.weight'.format(l)])
        #    for l in range(self.num_layers)])
        #self.conv_pb_rep = nn.ParameterList([nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_pw.bias'.format(l)])
        #    for l in range(self.num_layers)])

        self.conv_dw_rep = []
        self.conv_pw_rep = []
        self.conv_pb_rep = []
        for l in range(self.num_layers):
            setattr(self, "conv_dw{}".format(l), nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_dw.weight'.format(l)]))
            self.conv_dw_rep.append(getattr(self, "conv_dw{}".format(l)))
        for l in range(self.num_layers):
            setattr(self, "conv_pw{}".format(l), nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_pw.weight'.format(l)]))
            self.conv_pw_rep.append(getattr(self, "conv_pw{}".format(l)))
        for l in range(self.num_layers):
            setattr(self, "conv_pb{}".format(l), nn.Parameter(pretrain_init['class_net.conv_rep.{}.conv_pw.bias'.format(l)]))
            self.conv_pb_rep.append(getattr(self, "conv_pb{}".format(l)))


        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        self.running_mu = torch.zeros(num_channels).to('cuda')
        self.running_std = torch.ones(num_channels).to('cuda')
        self.act = Swish(inplace=True)

        self.predict_dw = nn.Parameter(pretrain_init['class_net.predict.conv_dw.weight'])
        self.predict_pw = nn.Parameter(torch.randn((num_anchors,num_channels,1,1))*((1/num_channels)**0.5))
        self.predict_pb = nn.Parameter(torch.full([num_anchors],-math.log((1 - 0.01) / 0.01)))
        self.predict = [self.predict_dw, self.predict_pw, self.predict_pb]
        #self.predict = nn.ParameterList([
        #    nn.Parameter(pretrain_init['class_net.predict.conv_dw.weight']),
        #    nn.Parameter(torch.randn((num_anchors,num_channels,1,1))*((1/num_channels)**0.5)),
        #    nn.Parameter(torch.full([num_anchors],-math.log((1 - 0.01) / 0.01)))])


        self.bn_rep_w = []
        self.bn_rep_b = []
        '''for _ in range(FLAGS.num_conv):
            self.bn_rep.append(nn.ParameterList([nn.ParameterDict({'w':nn.Parameter(torch.ones(num_channels)),
                'b':nn.Parameter(torch.zeros(num_channels))})
                for _ in range(self.num_levels)]))'''

        for lev in range(self.num_levels):
            for rep in range(self.num_layers):
                setattr(self, "bn_w{}{}".format(rep,lev), nn.Parameter(pretrain_init['class_net.bn_rep.{}.{}.bn.weight'.format(rep,lev)]))
                self.bn_rep_w.append(getattr(self, "bn_w{}{}".format(rep,lev)))

        for lev in range(self.num_levels):
            for rep in range(self.num_layers):
                setattr(self, "bn_b{}{}".format(rep,lev), nn.Parameter(pretrain_init['class_net.bn_rep.{}.{}.bn.bias'.format(rep,lev)]))
                self.bn_rep_b.append(getattr(self, "bn_b{}{}".format(rep,lev)))

    def add_head(self):
        self.predict_pw_sep = nn.Parameter(torch.randn((self.num_anchors,self.num_channels,1,1))*((1/self.num_channels)**0.5))
        self.predict_pb_sep = nn.Parameter(torch.full([self.num_anchors],-math.log((1 - 0.01) / 0.01)))
        self.predict_class = [self.predict_pw_sep, self.predict_pb_sep]

    def forward(self, x, fast_weights=None, ret_activs=False, level_offset=0, heads='anch'):
        if fast_weights is None:
            conv_dw_rep, conv_pw_rep, conv_pb_rep = self.conv_dw_rep, self.conv_pw_rep, self.conv_pb_rep
            bn_rep_w, bn_rep_b, predict = self.bn_rep_w, self.bn_rep_b, self.predict
            if heads=='both' and FLAGS.separate_head:
                predict_class = self.predict_class
        else:
            conv_dw_rep = fast_weights[:self.num_layers]
            conv_pw_rep = fast_weights[self.num_layers:2*self.num_layers]
            conv_pb_rep = fast_weights[2*self.num_layers:3*self.num_layers]
            predict = fast_weights[3*self.num_layers:3*self.num_layers+3]
            bn_rep_w = fast_weights[3*self.num_layers+3:3*self.num_layers+3 + self.num_layers*self.num_levels]
            bn_rep_b = fast_weights[3*self.num_layers+3 + self.num_layers*self.num_levels:]
            heads = 'class'
            
        outputs = []
        if heads=='both' and FLAGS.separate_head:
            class_outputs = []

        if ret_activs: activs = []
        for level in range(level_offset,len(x)):
            x_level = x[level]
            bn_w_lev = bn_rep_w[level*self.num_layers:(level+1)*self.num_layers]
            bn_b_lev = bn_rep_b[level*self.num_layers:(level+1)*self.num_layers]
            for conv_dw,conv_pw,conv_pb,bn_w,bn_b in zip(conv_dw_rep,conv_pw_rep,conv_pb_rep,bn_w_lev,bn_b_lev):
                x_level = F.pad(x_level,(1,1,1,1))
                x_level = F.conv2d(x_level, conv_dw, groups=conv_dw.shape[0], padding=(0,0))
                x_level = F.conv2d(x_level, conv_pw, bias=conv_pb)
                x_level = F.batch_norm(x_level,self.running_mu,self.running_std,bn_w,bn_b,training=True)
                x_level = self.act(x_level)

            #if ret_activs: activs.append(x_level)
            x_pred = F.pad(x_level,(1,1,1,1))
            x_pred = F.conv2d(x_pred, predict[0], groups=predict[0].shape[0])
            if ret_activs: activs.append(x_pred)
            x_out = F.conv2d(x_pred, predict[1], bias=predict[2])
            outputs.append(x_out)
            if heads=='both' and FLAGS.separate_head:
                class_outputs.append(F.conv2d(x_pred, predict_class[0], bias=predict_class[1]))

        if heads=='both' and FLAGS.separate_head:
            if ret_activs:
                return class_outputs,outputs,activs
            else:
                return class_outputs,outputs
        else:
            if ret_activs:
                return outputs,activs
            else:
                return outputs


class ProjectionNet(nn.Module):
    
    def __init__(self, config, width):
        super(ProjectionNet, self).__init__()

        self.dot_mult = nn.Parameter(torch.tensor(FLAGS.dot_mult))
        self.dot_add = nn.Parameter(torch.tensor(FLAGS.dot_add))

        locs = torch.arange(start=-1.,end=1.,step=1/8)*3.14159
        locs = locs[:9]
        anch_enc = []
        for freq in range(4):
            anch_enc.append(torch.sin(2**freq * locs))
            anch_enc.append(torch.cos(2**freq * locs))

        self.anch_enc = torch.stack(anch_enc).transpose(0,1).cuda()

        locs = torch.arange(start=-1.,end=1.,step=1/64)*3.14159
        locs = locs[:80]
        cell_enc = []
        for freq in range(7):
            cell_enc.append(torch.sin(2**freq * locs))
            cell_enc.append(torch.cos(2**freq * locs))

        self.cell_enc = torch.stack(cell_enc).transpose(0,1).cuda()

        locs = torch.arange(start=-1.,end=1.,step=1/4)*3.14159
        locs = locs[:5]
        lev_enc = []
        for freq in range(3):
            lev_enc.append(torch.sin(2**freq * locs))
            lev_enc.append(torch.cos(2**freq * locs))

        self.lev_enc = torch.stack(lev_enc).transpose(0,1).cuda()

        self.width = width
        if FLAGS.proj_depth == 2:
            self.projection = nn.Sequential(nn.Linear(config.fpn_channels+8+28+6, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, int(width/2), bias=False))
        elif FLAGS.proj_depth == 3:
            self.projection = nn.Sequential(nn.Linear(config.fpn_channels+8+28+6, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, int(width/2), bias=False))
        elif FLAGS.proj_depth == 4:
            self.projection = nn.Sequential(nn.Linear(config.fpn_channels+8+28+6, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, width, bias=False), nn.ReLU(),
                                            nn.Linear(width, int(width/2), bias=False))

    def weighted_median(self, embds, confs):
        conf_sum = confs.sum()
        sorted_elems, sorted_idxs = torch.sort(embds,dim=0)
        sorted_confs = confs[sorted_idxs.transpose(0,1)].transpose(0,1)
        cum_sum = torch.cumsum(sorted_confs,dim=0)
        mask = (cum_sum >= conf_sum/2).long()
        median_idxs = torch.argmax(mask, dim=0).view(1,-1)
        if FLAGS.median_grad:
            median_embd = torch.gather(sorted_elems,0,median_idxs)
        else:
            median_embd = torch.gather(sorted_elems,0,median_idxs).detach()

        return median_embd, conf_sum


    def forward(self, x):
        return self.projection(x)


class AnchorNet(nn.Module):

    def __init__(self, config, at_start=True):
        super(AnchorNet, self).__init__()
        self.config = config
        self.num_levels = config.num_levels
        if not FLAGS.supp_alpha:
            self.alpha = None
        else:
            if FLAGS.learn_alpha:
                self.alpha = nn.Parameter(torch.tensor(FLAGS.inner_alpha))
            else:
                self.alpha = FLAGS.inner_alpha

        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)

        num_channels = 88

        if FLAGS.num_anch_layers == 1:
            anchor_out_kwargs = dict(
            in_channels=config.fpn_channels, out_channels=9, kernel_size=3,
            padding=config.pad_type, bias=True, norm_layer=None, act_layer=None)
            self.conv_rep = nn.ModuleList()
        else:
            anchor_out_kwargs = dict(
            in_channels=num_channels, out_channels=9, kernel_size=3,
            padding=config.pad_type, bias=True, norm_layer=None, act_layer=None)

            in_conv = dict(
                in_channels=config.fpn_channels, out_channels=num_channels, kernel_size=3,
                padding=config.pad_type, bias=True, act_layer=None, norm_layer=None)
            conv_kwargs = dict(
                in_channels=num_channels, out_channels=num_channels, kernel_size=3,
                padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
            self.conv_rep = nn.ModuleList([SeparableConv2d(**in_conv)]+[SeparableConv2d(**conv_kwargs) for _ in range(FLAGS.num_anch_layers-2)])
        
        self.bn_rep = nn.ModuleList()
        for _ in range(FLAGS.num_anch_layers-1):
            self.bn_rep.append(nn.ModuleList([
                nn.Sequential(OrderedDict([('bn', norm_layer(num_channels))]))
                for _ in range(self.num_levels)]))

        self.act = Swish(inplace=True)

        self.anchor_out = SeparableConv2d(**anchor_out_kwargs)

        for n, m in self.named_modules():
            _init_weight(m, n)

    def forward(self, x):
        outputs = []
        for level in range(len(x)):
            if FLAGS.detach_anch:
                x_level = x[level].detach()
            else:
                x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)
                x_level = self.act(x_level)
            outputs.append(self.anchor_out(x_level))
        return outputs


class EfficientDet(nn.Module):

    def __init__(self, config, pretrained_backbone=True, alternate_init=False):
        super(EfficientDet, self).__init__()
        self.config = config
        #set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name, features_only=True, out_indices=(2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(self.config, feature_info)
        self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
        self.box_net = HeadNet(self.config, num_outputs=4)
        self.num_anchors = len(config.aspect_ratios) * config.num_scales

        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    @torch.jit.ignore()
    def reset_head(self, num_classes=None, num_channels=None, aspect_ratios=None, num_scales=None, alternate_init=False):
        reset_class_head = False
        reset_box_head = False
        set_config_writeable(self.config)
        if num_classes is not None:
            reset_class_head = True
            self.config.num_classes = num_classes
        if aspect_ratios is not None:
            reset_box_head = True
            self.config.aspect_ratios = aspect_ratios
        if num_scales is not None:
            reset_box_head = True
            self.config.num_scales = num_scales
        set_config_readonly(self.config)

        if reset_class_head:
            self.class_net.predict.conv_pw = create_conv2d(self.config.fpn_channels, num_classes*self.num_anchors, 1, padding='', bias=True)
            self.class_net.predict.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))

            '''self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes, num_channels_flag=num_channels)
            for n, m in self.class_net.named_modules(prefix='class_net'):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)'''

        if reset_box_head:
            self.box_net = HeadNet(self.config, num_outputs=4)
            for n, m in self.box_net.named_modules(prefix='box_net'):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    @torch.jit.ignore()
    def toggle_head_bn_level_first(self):
        """ Toggle the head batchnorm layers between being access with feature_level first vs repeat
        """
        self.class_net.toggle_bn_level_first()
        self.box_net.toggle_bn_level_first()

    def forward(self, x, fast_weights=None, ret_activs=False, mode='full_net'):
        if mode=='supp_cls':
            return self.class_net(x,fast_weights=fast_weights,ret_activs=True,level_offset=FLAGS.supp_level_offset, heads='both')
        elif mode=='supp_bb':
            x = self.backbone(x)
            activs = self.fpn(x)
            return activs
        elif mode=='bb':
            feats = self.backbone(x)
            return feats
        elif mode=='not_cls':
            activs = self.fpn([feat.to('cuda:0') for feat in x])
            x_box = self.box_net([activ.to('cuda') for activ in activs])
            return activs, x_box
        elif mode=='qry_cls':
            x_class = self.class_net(x,fast_weights=fast_weights, ret_activs=ret_activs, heads='None')
            return x_class
        elif mode=='full_net':
            x = self.backbone(x)
            activs = self.fpn(x)
            x_class = self.class_net(activs)
            x_box = self.box_net(activs)
            return x_class, x_box
        elif mode=='head':
            x_class = self.class_net(x)
            x_box = self.box_net(x)
            return x_class, x_box
        elif mode=='only_fpn':
            activs = self.fpn(x)
            return activs
        elif mode=='fpn':
            feats = self.backbone(x)
            activs = self.fpn(feats)
            return feats,activs
        elif mode=='fpn_and_head':
            activs = self.fpn(x)
            x_class = self.class_net(activs)
            x_box = self.box_net(activs)
            return x_class,x_box
        


        '''if mode=='fpn':
            feats = self.backbone(x)
            activs = self.fpn(feats)
            return feats,activs
        elif mode=='support':
            activs, x_class = self.class_net(x,ret_activs=True)
            lab_mult = []
            for level in range(self.num_levels):
                lab_mult.append(self.anchor_net(activs[level]).sigmoid())
            return x_class,lab_mult
        elif mode=='head':
            x_class = self.class_net(x)
            x_box = self.box_net(x)
            return x_class, x_box
        elif mode=='gen_activ':
            x = self.backbone(x)
            x = self.fpn(x)
            return x
        elif mode=='full_net':
            x = self.backbone(x)
            x = self.fpn(x)
            x_class = self.class_net(x)
            x_box = self.box_net(x)
            return x_class, x_box'''




