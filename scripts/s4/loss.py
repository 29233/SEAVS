import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from einops import rearrange


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    first_gt_mask = first_gt_mask.unsqueeze(1)  # [bs, 1, 224, 224]
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss


def F1_Dice_loss(pred_masks, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs*5, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, 1, h, w)
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    pred_mask = first_pred.flatten(1)
    gt_mask = first_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def IouSemanticAwareLoss(pred_masks, mask_feature, gt_mask, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}

    if loss_type == 'bce':
        loss_func = F1_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F1_Dice_loss
    else:
        raise ValueError

    iou_loss = loss_func(pred_masks, gt_mask)
    total_loss += weight_dict['iou_loss'] * iou_loss
    loss_dict['iou_loss'] = weight_dict['iou_loss'] * iou_loss.item()

    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    mix_loss = weight_dict['mix_loss']*loss_func(mask_feature, gt_mask)
    total_loss += mix_loss
    loss_dict['mix_loss'] = mix_loss.item()

    return total_loss, loss_dict

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs[0][0], targets[0], reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).mean()

def F1_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    targets = targets.unsqueeze(1)
    indices = torch.tensor(list(range(0, len(inputs), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(inputs, dim=0, index=indices)
    prob = first_pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(first_pred, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()

def F1_Feature_Align_loss(audio_context, visual_context, t=0.05):
    """
    feature align loss for audio and visual features

    Args:
    audio_feat: audio feature, shape: [bst, c]
    visual_feat: visual feature, shape: [bst, c]
    """
    # assert len(audio_context.shape) == 2
    # assert len(visual_context.shape) == 2
    # bst, c = visual_context.shape
    # T = 5
    # bs = bst // T
    # targets = torch.arange(bs).to(visual_context.device)
    # # target = torch.arange(bst).to(visual_context.device)
    # visual_context = visual_context / visual_context.norm(dim=1, keepdim=True)
    # audio_context = audio_context / audio_context.norm(dim=1, keepdim=True)
    #
    # audio_context = rearrange(audio_context, '(b t) c -> t b c', b=bs, t=T)[0]
    # visual_context = rearrange(visual_context, '(b t) c -> t b c', b=bs, t=T)[0]
    #
    # scale_factor = np.exp(t)
    # logits_a = torch.einsum('Bc,bc->Bb', audio_context, visual_context) * scale_factor
    # logits_v = torch.einsum('Bc,bc->Bb', visual_context, audio_context) * scale_factor
    #
    # # context_align = torch.matmul(audio_context, visual_context.t())
    # # context_align = F.softmax(context_align * scale_factor, dim=1)
    # # loss = F.cross_entropy(context_align * scale_factor, target)
    #
    # loss = (F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_v, targets)) / 2
    # return loss
    assert len(audio_context.shape) == 2
    assert len(visual_context.shape) == 2
    bst, c = visual_context.shape
    T = 5
    bs = bst // T
    targets = torch.arange(bs).unsqueeze(0).repeat(T, 1).to(visual_context.device)
    # target = torch.arange(bst).to(visual_context.device)
    visual_context = visual_context / visual_context.norm(dim=1, keepdim=True)
    audio_context = audio_context / audio_context.norm(dim=1, keepdim=True)

    audio_context = rearrange(audio_context, '(b t) c -> t b c', b=bs, t=T)
    visual_context = rearrange(visual_context, '(b t) c -> t b c', b=bs, t=T)

    scale_factor = np.exp(t)
    logits_a = torch.einsum('tBc,tbc->tBb', audio_context, visual_context) * scale_factor
    logits_v = torch.einsum('tBc,tbc->tBb', visual_context, audio_context) * scale_factor

    loss = (F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_v, targets)) / 2

    return loss

def Loss(pred_mask, gt_mask, weight_dict):
    # pred_mask: l, bs * t, 1, h, w          # 改为直接用dice_loss和focal_loss的
    # gt_mask: bs x t, 1, h, w

    total_loss = 0
    dice_loss = F1_Dice_loss(pred_mask, gt_mask)
    mask_loss = F1_sigmoid_focal_loss(pred_mask, gt_mask)

    total_loss = weight_dict['dice_loss'] * dice_loss + weight_dict['focal_loss'] * mask_loss

    loss_dict = {'focal_loss': mask_loss.item(), 'dice_loss': dice_loss.item(), 'total_loss': total_loss.item()}

    return total_loss, loss_dict


def Context_align_Loss(pred_mask, gt_mask, audio_context, visual_context, weight_dict):
    # pred_mask: l, bs * t, 1, h, w          # 改为直接用dice_loss和focal_loss的
    # gt_mask: bs x t, 1, h, w

    total_loss = 0
    dice_loss = F1_Dice_loss(pred_mask, gt_mask)
    mask_loss = F1_IoU_BCELoss(pred_mask, gt_mask)
    context_loss = F1_Feature_Align_loss(audio_context, visual_context)

    total_loss = weight_dict['dice_loss'] * dice_loss + weight_dict['focal_loss'] * mask_loss + weight_dict['context_loss'] * context_loss

    loss_dict = {'focal_loss': mask_loss.item(), 'dice_loss': dice_loss.item(), 'context_loss': context_loss.item() , 'total_loss': total_loss.item()}

    return total_loss, loss_dict