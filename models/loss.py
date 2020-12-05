import torch
import torch.nn as nn
import torch.nn.functional as F

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return all_losses

# def model_loss(disp_ests, disp_probs, disp_gt, mask, weights = [0.5, 0.5, 0.7, 1.0, 1.0], maxdisp=192):
#     all_losses = []
#     disp_ocr = disp_ests[-1]
#     all_losses.append(weights[0] * F.smooth_l1_loss(disp_ocr[mask], disp_gt[mask], size_average=True))
#
#     maxdisp = maxdisp // 4
#     CEloss = nn.CrossEntropyLoss()
#     disp_gt = (F.interpolate(disp_gt.unsqueeze(1), scale_factor=0.25) // 4).squeeze(1)
#     zeros = torch.zeros_like(disp_gt)
#     gt_probs = torch.where((disp_gt >= maxdisp) | (disp_gt <= 0), zeros, disp_gt)
#     mask_new = ((disp_gt >= maxdisp) | (disp_gt <= 0)).unsqueeze(1).repeat(1,48,1,1)
#     zeros = torch.zeros_like(disp_probs[0])
#     gt_probs = gt_probs.to(dtype=torch.long)
#     for disp_prob, weight in zip(disp_probs, weights[1:]):
#         disp_prob = torch.where(mask_new, zeros, disp_prob)
#         all_losses.append(weight * CEloss(disp_prob, gt_probs))
#     return all_losses
