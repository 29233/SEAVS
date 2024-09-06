import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).mean()


def F5_Dice_loss(pred_mask, five_gt_masks):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def F5_Feature_Align_loss(audio_context, visual_context, t=0.05):
    """
    feature align loss for audio and visual features

    Args:
    audio_feat: audio feature, shape: [bst, c]
    visual_feat: visual feature, shape: [bst, c]
    """
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

    # context_align = torch.matmul(audio_context, visual_context.t())
    # context_align = F.softmax(context_align * scale_factor, dim=1)
    # loss = F.cross_entropy(context_align * scale_factor, target)

    loss = (F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_v, targets)) / 2

    return loss


def IouSemanticAwareLoss(pred_mask, gt_mask, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}

    if loss_type == 'bce':
        loss_func = F5_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F5_Dice_loss
    else:
        raise ValueError

    iou_loss = weight_dict['iou_loss'] * loss_func(pred_mask, gt_mask)
    total_loss += iou_loss
    loss_dict['iou_loss'] = iou_loss.item()

    mask_loss = sigmoid_focal_loss(pred_mask, gt_mask)

    total_loss = weight_dict['iou_loss'] * iou_loss + weight_dict['mask_loss'] * mask_loss

    # mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    # mask_feature = F.interpolate(
    #     mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    # mix_loss = weight_dict['mix_loss']*loss_func(mask_feature, gt_mask)
    # total_loss += mix_loss
    # loss_dict['mix_loss'] = mix_loss.item()
    loss_dict['mask_loss'] = mask_loss.item()
    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict


def Context_align_Loss(pred_mask, gt_mask, audio_context, visual_context, weight_dict):
    # pred_mask: l, bs * t, 1, h, w          # 改为直接用dice_loss和focal_loss的
    # gt_mask: bs x t, 1, h, w

    total_loss = 0
    dice_loss = F5_Dice_loss(pred_mask, gt_mask)
    mask_loss = F5_IoU_BCELoss(pred_mask, gt_mask)
    context_loss = F5_Feature_Align_loss(audio_context, visual_context)

    total_loss = weight_dict['dice_loss'] * dice_loss + weight_dict['focal_loss'] * mask_loss + weight_dict[
        'context_loss'] * context_loss

    loss_dict = {'focal_loss': mask_loss.item(), 'dice_loss': dice_loss.item(), 'context_loss': context_loss.item(),
                 'total_loss': total_loss.item()}

    return total_loss, loss_dict


def Loss(pred_mask, gt_mask, weight_dict):
    # pred_mask: l, bs * t, 1, h, w          # 改为直接用dice_loss和focal_loss的
    # gt_mask: bs x t, 1, h, w

    total_loss = 0
    dice_loss = F5_Dice_loss(pred_mask, gt_mask)
    mask_loss = F5_IoU_BCELoss(pred_mask, gt_mask)

    total_loss = weight_dict['dice_loss'] * dice_loss + weight_dict['focal_loss'] * mask_loss

    loss_dict = {'focal_loss': mask_loss.item(), 'dice_loss': dice_loss.item(), 'total_loss': total_loss.item()}

    return total_loss, loss_dict


# class ClipLoss(nn.Module):
#
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod
#
#         # cache state
#         self.prev_num_logits = 0
#         self.labels = {}
#
#     def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#         # calculated ground-truth and cache if enabled
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]
#         return labels
#
#     def get_logits(self, image_features, text_features, logit_scale):
#         if self.world_size > 1:
#             all_image_features, all_text_features = gather_features(
#                 image_features, text_features,
#                 self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
#
#             if self.local_loss:
#                 logits_per_image = logit_scale * image_features @ all_text_features.T
#                 logits_per_text = logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = logit_scale * image_features @ text_features.T
#             logits_per_text = logit_scale * text_features @ image_features.T
#
#         return logits_per_image, logits_per_text
#
#     def forward(self, image_features, text_features, logit_scale, output_dict=False):
#         device = image_features.device
#         logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
#
#         labels = self.get_ground_truth(device, logits_per_image.shape[0])
#
#         total_loss = (
#                              F.cross_entropy(logits_per_image, labels) +
#                              F.cross_entropy(logits_per_text, labels)
#                      ) / 2
#
#         return {"contrastive_loss": total_loss} if output_dict else total_loss
