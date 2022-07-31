from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def lovasz_grad(gt_sorted: List):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(
    preds: torch.Tensor, labels: torch.Tensor, EMPTY: float = 1.0, ignore: Any = None, per_image: bool = True
):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(
    preds: torch.Tensor,
    labels: torch.Tensor,
    C: int,
    EMPTY: float = 1.0,
    ignore: Any = None,
    per_image: bool = False,
):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious))  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


class TScore(nn.Module):
    def __init__(self, coefficient: float = 2.5, eps: float = 0.001):
        super().__init__()

        self.coefficient = torch.tensor(coefficient)
        self.eps = torch.tensor(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r"""
        Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id

        The loss function is defined by:

        $score =  sum[1-(1 - flatten(outputs)*flatten(targets))^coefficient]
                \[sum[1-((1 - flatten(outputs))*(1-flatten(targets)))^coefficient]]$
        and
        loss = 1-score
        """
        return t_score_loss(logits, targets, self.coefficient, self.eps)


def t_score_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    r"""
    Binary Lovasz hinge loss
    logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    per_image: compute the loss per image instead of per batch
    ignore: void class id

    The loss function is defined by:

    $score =  sum[1-(1 - flatten(outputs)*flatten(targets))^coefficient]
            \[sum[1-((1 - flatten(outputs))*(1-flatten(targets)))^coefficient]]$
    and
    loss = 1-score
    """
    coefficient = 2.5
    eps = 0.0001
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(logits)}")

    if not len(logits.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {logits.shape}")

    if not logits.shape[-2:] == targets.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {logits.shape} and {targets.shape}")

    if not logits.device == targets.device:
        raise ValueError(f"input and target must be in the same device. Got: {logits.device} and {targets.device}")
    outputs = torch.sigmoid(logits)

    outputs = outputs.contiguous().view(-1, 1)
    targets = targets.contiguous().view(-1, 1)

    y1_coefficient = torch.mul(outputs, targets)
    y2_coefficient = torch.mul(torch.tensor(1.0) - outputs, torch.tensor(1.0) - targets)
    f1_coefficient = torch.tensor(1.0) - (torch.tensor(1.0) - y1_coefficient) ** coefficient
    f2_coefficient = (torch.tensor(1.0) - y2_coefficient) ** coefficient
    score = torch.sum(f1_coefficient) / (torch.sum(f2_coefficient) + eps)
    loss = torch.tensor(1.0) - score
    return loss


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = True, ignore: Any = None):
    r"""
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor):
    r"""
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor, ignore: Any = None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits: torch.Tensor, labels: torch.Tensor, ignore: Any = None):
    r"""
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    only_present: bool = False,
    per_image: bool = False,
    ignore: Any = None,
):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                only_present=only_present,
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, only_present: bool = False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore: Any = None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits: torch.Tensor, labels: torch.Tensor, ignore: Any = None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------


def mean(list_: List, ignore_nan: bool = False, empty: int = 0):
    """
    nanmean compatible with generators.
    """
    list_ = iter(list_)
    if ignore_nan:
        list_ = ifilterfalse(np.isnan, list_)
    try:
        n = 1
        acc = next(list_)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(list_, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def symmetric_lovasz(outputs: torch.Tensor, targets: torch.Tensor):
    return 0.5 * (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


def soft_jaccard(outputs: torch.Tensor, targets: torch.Tensor):
    eps = 1e-15
    jaccard_target = (targets == 1).float()
    jaccard_output = F.sigmoid(outputs)

    intersection = (jaccard_output * jaccard_target).sum()
    union = jaccard_output.sum() + jaccard_target.sum()
    return intersection / (union - intersection + eps)


def binary_dice_coefficient(
    logit: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    computes the dice coefficient for a binary segmentation task

    Args:
        logit: predicted segmentation (of shape Nx(Dx)HxW)
        gt: target segmentation (of shape NxCx(Dx)HxW)
        thresh: segmentation threshold
        smooth: smoothing value to avoid division by zero

    Returns:
        torch.Tensor: dice score
    """
    thresh = 0
    smooth = 1e-7

    assert logit.shape == gt.shape

    pred_bool = logit > thresh

    intersec = (pred_bool * gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum() + gt.float().sum() + smooth)


class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)
    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight: float = 0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            loss += self.jaccard_weight * (1 - soft_jaccard(outputs, targets))
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma: float = 2, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit: torch.Tensor, target: torch.Tensor, class_weight=None):
        target = target.view(-1, 1).long()

        if class_weight is None:
            class_weight = [1] * 2  # [0.5, 0.5]

        prob = F.sigmoid(logit)
        prob = prob.view(-1, 1)
        prob = torch.cat((1 - prob, prob), 1)
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.0)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = -class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class PseudoBCELoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        z = logit.view(-1)
        t = truth.view(-1)
        loss = z.clamp(min=0) - z * t + torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / len(t)  # w.sum()
        return loss


def calc_iou(actual: np.ndarray, pred: np.ndarray):
    intersection = np.count_nonzero(actual * pred)
    union = np.count_nonzero(actual) + np.count_nonzero(pred) - intersection
    iou_result = intersection / union if union != 0 else 0.0
    return iou_result


def calc_ious(actuals: np.ndarray, preds: np.ndarray):
    ious_ = np.array([calc_iou(a, p) for a, p in zip(actuals, preds)])
    return ious_


def calc_precisions(thresholds: float, ious: np.ndarray) -> np.ndarray:
    thresholds = np.reshape(thresholds, (1, -1))
    ious = np.reshape(ious, (-1, 1))
    ps = ious > thresholds
    mps = ps.mean(axis=1)
    return mps


def indiv_scores(masks: np.ndarray, preds: np.ndarray):
    masks[masks > 0] = 1
    preds[preds > 0] = 1
    ious = calc_ious(masks, preds)
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = calc_precisions(thresholds, ious)

    # Adjust score for empty masks
    emptyMasks = np.count_nonzero(masks.reshape((len(masks), -1)), axis=1) == 0
    emptyPreds = np.count_nonzero(preds.reshape((len(preds), -1)), axis=1) == 0
    adjust = (emptyMasks == emptyPreds).astype(np.float)
    precisions[emptyMasks] = adjust[emptyMasks]

    return precisions


def calc_metric(masks: np.ndarray, preds: np.ndarray):
    return np.mean(indiv_scores(masks, preds))


def do_kaggle_metric(predict: Union[np.ndarray, torch.Tensor], truth: Union[np.ndarray, torch.Tensor], threshold=0.5):
    """
    input includes 3 parametters:
     predict:  x in (-infty,+infty)
     truth  :  y in (0,1)
     threshold
    """
    EPS = 1e-12
    N = len(predict)
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    predict = predict > threshold
    truth = truth > 0.5
    intersection = truth & predict
    union = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + EPS)

    # -------------------------------------------
    result = []
    precision = []
    is_empty_truth = truth.sum(1) == 0
    is_empty_predict = predict.sum(1) == 0

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou >= t

        tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
        fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
        fn = (~is_empty_truth) & (is_empty_predict)
        fp_empty = (is_empty_truth) & (~is_empty_predict)
        tn_empty = (is_empty_truth) & (is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None):
        target = target.view(-1, 1).long()

        if class_weight is None:
            class_weight = [1] * 2  # [0.5, 0.5]

        prob = F.sigmoid(logit)
        prob = prob.view(-1, 1)
        prob = torch.cat((1 - prob, prob), 1)
        select = torch.FloatTensor(len(prob), 2).zero_().cuda()
        select.scatter_(1, target, 1.0)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = -class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


def soft_dice_loss(outputs, targets, per_image=False, per_channel=False):
    batch_size, n_channels = outputs.size(0), outputs.size(1)

    eps = 1e-6
    n_parts = 1
    if per_image:
        n_parts = batch_size
    if per_channel:
        n_parts = batch_size * n_channels

    dice_target = targets.contiguous().view(n_parts, -1).float()
    dice_output = outputs.contiguous().view(n_parts, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def dice_metric(preds, trues, per_image=False, per_channel=False):
    preds = preds.float()
    return 1 - soft_dice_loss(preds, trues, per_image, per_channel)


EPSILON = 1e-15


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


def jaccard(
    outputs: torch.Tensor, targets: torch.Tensor, per_image: bool = False, non_empty: bool = False, min_pixels: int = 5
):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty is True:
        assert per_image is True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images

    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight: float = None, size_average: bool = True, per_image: bool = False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer("weight", weight)
        self.per_image = per_image

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(nn.Module):
    def __init__(
        self,
        weight: float = None,
        size_average: bool = True,
        per_image: bool = False,
        non_empty: bool = False,
        apply_sigmoid: bool = False,
        min_pixels: int = 5,
    ):
        super().__init__()
        self.size_average = size_average
        self.register_buffer("weight", weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(
            input,
            target,
            per_image=self.per_image,
            non_empty=self.non_empty,
            min_pixels=self.min_pixels,
        )


class StableBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        # todo check correctness
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class ComboLoss(nn.Module):
    def __init__(
        self,
        weights,
        per_image=False,
        channel_weights=[1, 0.5, 0.5],
        channel_losses=None,
    ):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {
            "bce": self.bce,
            "dice": self.dice,
            "focal": self.focal,
            "jaccard": self.jaccard,
            "lovasz": self.lovasz,
            "lovasz_sigmoid": self.lovasz_sigmoid,
        }
        self.expect_sigmoid = {"dice", "focal", "jaccard", "lovasz_sigmoid"}
        self.per_channel = {"dice", "jaccard", "lovasz_sigmoid"}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...],
                        )

            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss.clamp(min=1e-5)


class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return symmetric_lovasz(outputs, targets)


class LovaszLossSigmoid(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_sigmoid(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class DiceBCELoss(nn.Module):
    # Formula Given above.
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)
        inputs = F.softmax(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = 0.5 * BCE + 0.5 * dice_loss

        return Dice_BCE


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def tversky_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        input: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = tversky_loss(input, target, alpha=0.5, beta=0.5)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {input.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    fps = torch.sum(input_soft * (-target_one_hot + 1.0), dims)
    fns = torch.sum((-input_soft + 1.0) * target_one_hot, dims)

    numerator = intersection
    denominator = intersection + alpha * fps + beta * fns
    tversky_loss = numerator / (denominator + eps)

    return torch.mean(-tversky_loss + 1.0)


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coefficient loss.

    According to :cite:`salehi2017tversky`, we compute the Tversky Coefficient as follows:

    .. math::

        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \setminus G| + \beta |G \setminus P|}

    Where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.

    Note:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff

    Args:
        alpha: the first coefficient in the denominator.
        beta: the second coefficient in the denominator.
        eps: scalar for numerical stability.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> criterion = TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, beta: float, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return tversky_loss(input, target, self.alpha, self.beta, self.eps)


if __name__ == "__main__":
    loss_function = TScore(coefficient=2.5)
    preds = torch.rand((2, 3, 384, 384))
    targets = torch.rand((2, 3, 384, 384))
    t_loss = loss_function(preds, targets)
    print(f"T loss: {t_loss}")
