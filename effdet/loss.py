""" EfficientDet Focal, Huber/Smooth L1 loss fns w/ jit support

Based on loss fn in Google's automl EfficientDet repository (Apache 2.0 license).
https://github.com/google/automl/tree/master/efficientdet

Copyright 2020 Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple


def focal_loss_legacy(logits, targets, alpha: float, gamma: float, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    'Legacy focal loss matches the loss used in the official Tensorflow impl for initial
    model releases and some time after that. It eventually transitioned to the 'New' loss
    defined below.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))

    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    return weighted_loss / normalizer

def new_focal_loss(logits, targets, alpha: float, gamma: float, normalizer, label_smoothing: float = 0.01, loss_func=F.binary_cross_entropy_with_logits):
    """Compute the focal loss between `logits` and the golden `target` values.

    'New' is not the best descriptor, but this focal loss impl matches recent versions of
    the official Tensorflow impl of EfficientDet. It has support for label smoothing, however
    it is a bit slower, doesn't jit optimize well, and uses more memory.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].
        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.
        gamma: A float32 scalar modulating loss from hard and easy examples.
        normalizer: Divide loss by this value.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    Returns:
        loss: A float32 scalar representing normalized total loss.
    """
    # compute focal loss multipliers before label smoothing, such that it will not blow up the loss.
    #print(logits.max())

    
    targets = targets.to(logits.dtype)
    if not alpha is None:
        pred_prob = logits.sigmoid()
        onem_targets = 1. - targets
        #p_t = (targets * pred_prob) + (onem_targets * (1. - pred_prob))
        alpha_factor = targets * alpha + onem_targets * (1. - alpha)
        #modulating_factor = torch.pow(1. - p_t, gamma)

    #else:
    #    targets = targets.detach()

    # apply label smoothing for cross_entropy for each entry.
    if label_smoothing > 0.:
        targets = targets * (1. - label_smoothing) + .5 * label_smoothing

    loss = loss_func(logits, targets, reduction='none')
    #loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    if not alpha is None:
        # compute the final loss and return
        return (1 / normalizer) * alpha_factor * loss
    else:
        return (1 / normalizer) * loss

def cosine_loss(input, target, margin=0., reduction='mean'):
    mask = target == 1.
    loss = torch.where(mask, 1-input, input-margin)
    loss[loss < 0.] = 0.
    return loss.mean()


def huber_loss(
        input, target, delta: float = 1., weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def smooth_l1_loss(
        input, target, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = False):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = input - target
        abs_err = torch.abs(err)
        '''sort,_ = torch.sort(abs_err.reshape(-1))
        print(sort[:10],sort[20],sort[50],sort[100],sort[800])
        conf_err = torch.where(weights > beta, abs_err, torch.tensor(1000.,dtype=torch.float32, device='cuda'))
        print(conf_err[conf_err != 1000.].mean(),conf_err[conf_err != 1000.].max())
        sort_conf,_ = torch.sort(conf_err.reshape(-1))
        print(sort_conf[:10],sort_conf[20],sort_conf[50],sort_conf[100],sort_conf[800])'''
        loss = torch.where(abs_err < beta, 0.5 * abs_err.pow(2) / beta, abs_err - 0.5 * beta)


    if weights is not None:
        loss *= weights
        weighted_sign = torch.sign(err)*weights
        pos_grad_sum = weighted_sign[weighted_sign > 0.].sum()
        neg_grad_sum = weighted_sign[weighted_sign < 0.].sum()

    if size_average:
        return loss.mean()
    else:
        return loss.sum(), pos_grad_sum, neg_grad_sum

def l2_loss(
        input, target, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = False):

    err = input - target
    loss = err**2

    if weights is not None:
        loss *= weights
        weighted_sign = torch.sign(err)*weights
        pos_grad_sum = weighted_sign[weighted_sign > 0.].sum()
        neg_grad_sum = weighted_sign[weighted_sign < 0.].sum()

    return loss.mean(), pos_grad_sum, neg_grad_sum


def _box_loss(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = huber_loss(box_outputs, box_targets, weights=mask, delta=delta, size_average=False)
    return box_loss / normalizer


def one_hot(x, num_classes: int):
    # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
    x_non_neg = (x >= 0).unsqueeze(-1)
    onehot = torch.zeros(x.shape + (num_classes,), device=x.device, dtype=torch.float32)
    return onehot.scatter(-1, x.unsqueeze(-1) * x_non_neg, 1) * x_non_neg

def class_loss_fn(
        cls_outputs: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        num_positives: torch.Tensor,
        num_classes: int,
        alpha: float,
        gamma: float,
        label_smoothing: float = 0.,
        legacy_focal: bool = False,
        loss_func=F.binary_cross_entropy_with_logits) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    num_positives_sum = (num_positives.sum() + 1.0)
    levels = len(cls_outputs)

    cls_losses = []
    for l in range(levels):
        #cls_targets_at_level = cls_targets[l]
        # Onehot encoding for classification labels.
        #cls_targets_at_level_oh = one_hot(cls_targets_at_level, num_classes)
        #bs, height, width, _, _ = cls_targets_at_level_oh.shape
        #cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)

        bs, _, height, width = cls_targets[l].shape
        cls_targets_at_level = cls_targets[l].permute(0, 2, 3, 1)
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1)
        cls_loss = new_focal_loss(
                cls_outputs_at_level, cls_targets_at_level,
                alpha=alpha, gamma=gamma, normalizer=num_positives_sum, label_smoothing=label_smoothing, loss_func=loss_func)
        cls_loss = cls_loss.reshape(bs, height, width, -1, num_classes)
        cls_losses.append(cls_loss.sum())   # FIXME reference code added a clamp here at some point ...clamp(0, 2))

    # Sum per level losses to total loss.
    cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
    return cls_loss


def loss_fn(
        cls_outputs: List[torch.Tensor],
        box_outputs: List[torch.Tensor],
        cls_targets: List[torch.Tensor],
        box_targets: List[torch.Tensor],
        num_positives: torch.Tensor,
        num_classes: int,
        alpha: float,
        gamma: float,
        delta: float,
        box_loss_weight: float,
        label_smoothing: float = 0.,
        legacy_focal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
            at each feature level (index)

        box_outputs: a List with values representing box regression targets in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)

        cls_targets: groundtruth class targets.

        box_targets: groundtrusth box targets.

        num_positives: num positive grountruth anchors

    Returns:
        total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

        cls_loss: an integer tensor representing total class loss.

        box_loss: an integer tensor representing total box regression loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = (num_positives.sum() + 1.0)
    levels = len(cls_outputs)

    cls_losses = []
    box_losses = []
    for l in range(levels):
        cls_targets_at_level = cls_targets[l]
        box_targets_at_level = box_targets[l]

        # Onehot encoding for classification labels.
        cls_targets_at_level_oh = one_hot(cls_targets_at_level, num_classes)

        bs, height, width, _, _ = cls_targets_at_level_oh.shape
        cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1)
        if legacy_focal:
            cls_loss = focal_loss_legacy(
                cls_outputs_at_level, cls_targets_at_level_oh,
                alpha=alpha, gamma=gamma, normalizer=num_positives_sum)
        else:
            cls_loss = new_focal_loss(
                cls_outputs_at_level, cls_targets_at_level_oh,
                alpha=alpha, gamma=gamma, normalizer=num_positives_sum, label_smoothing=label_smoothing)
        cls_loss = cls_loss.view(bs, height, width, -1, num_classes)
        cls_loss = cls_loss * (cls_targets_at_level != -2).unsqueeze(-1)
        cls_losses.append(cls_loss.sum())   # FIXME reference code added a clamp here at some point ...clamp(0, 2))

        box_losses.append(_box_loss(
            box_outputs[l].permute(0, 2, 3, 1),
            box_targets_at_level,
            num_positives_sum,
            delta=delta))

    # Sum per level losses to total loss.
    cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
    box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
    total_loss = cls_loss + box_loss_weight * box_loss
    return total_loss, cls_loss, box_loss


#loss_jit = torch.jit.script(loss_fn)

def box_only_loss(
        box_outputs: List[torch.Tensor],
        box_targets: List[torch.Tensor],
        num_positives: torch.Tensor,
        alpha: float,
        gamma: float,
        delta: float,
        box_loss_weight: float,
        label_smoothing: float = 0.,
        legacy_focal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
            at each feature level (index)

        box_outputs: a List with values representing box regression targets in
            [batch_size, height, width, num_anchors * 4] at each feature level (index)

        cls_targets: groundtruth class targets.

        box_targets: groundtrusth box targets.

        num_positives: num positive grountruth anchors

    Returns:
        total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

        cls_loss: an integer tensor representing total class loss.

        box_loss: an integer tensor representing total box regression loss.
    """
    # Sum all positives in a batch for normalization and avoid zero
    # num_positives_sum, which would lead to inf loss during training
    num_positives_sum = (num_positives.sum() + 1.0)
    levels = len(box_outputs)

    box_losses = []
    for l in range(levels):
        box_targets_at_level = box_targets[l]

        box_losses.append(_box_loss(
            box_outputs[l].permute(0, 2, 3, 1),
            box_targets_at_level,
            num_positives_sum,
            delta=delta))

    box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
    total_loss = box_loss_weight * box_loss
    return total_loss


class DetectionLoss(nn.Module):

    __constants__ = ['num_classes']

    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight
        self.label_smoothing = config.label_smoothing
        self.legacy_focal = config.legacy_focal
        self.use_jit = config.jit_loss

    def box_loss(
            self,
            box_outputs: List[torch.Tensor],
            box_targets: List[torch.Tensor],
            num_positives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_fn = box_only_loss

        return l_fn(
            box_outputs, box_targets, num_positives,
            alpha=self.alpha, gamma=self.gamma, delta=self.delta,
            box_loss_weight=self.box_loss_weight, label_smoothing=self.label_smoothing, legacy_focal=self.legacy_focal)

    def forward(
            self,
            cls_outputs: List[torch.Tensor],
            box_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor],
            box_targets: List[torch.Tensor],
            num_positives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_fn = loss_fn
        #if not torch.jit.is_scripting() and self.use_jit:
        #    # This branch only active if parent / bench itself isn't being scripted
        #    # NOTE: I haven't figured out what to do here wrt to tracing, is it an issue?
        #    l_fn = loss_jit

        return l_fn(
            cls_outputs, box_outputs, cls_targets, box_targets, num_positives,
            num_classes=self.num_classes, alpha=self.alpha, gamma=self.gamma, delta=self.delta,
            box_loss_weight=self.box_loss_weight, label_smoothing=self.label_smoothing, legacy_focal=self.legacy_focal)


class SupportLoss(nn.Module):

    __constants__ = ['num_classes']

    def __init__(self, config, loss_type):
        super(SupportLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.label_smoothing = config.label_smoothing
        self.legacy_focal = config.legacy_focal
        self.use_jit = config.jit_loss

        if loss_type == 'ce':
            self.loss_func = F.binary_cross_entropy_with_logits
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss

    def forward(
            self,
            cls_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor],
            num_positives: torch.Tensor,
            alpha) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_fn = class_loss_fn
        if not torch.jit.is_scripting() and self.use_jit:
            # This branch only active if parent / bench itself isn't being scripted
            # NOTE: I haven't figured out what to do here wrt to tracing, is it an issue?
            l_fn = loss_jit

        return l_fn(
            cls_outputs, cls_targets, num_positives,
            num_classes=self.num_classes, alpha=alpha, gamma=self.gamma,
            label_smoothing=self.label_smoothing, legacy_focal=self.legacy_focal, loss_func=self.loss_func)
