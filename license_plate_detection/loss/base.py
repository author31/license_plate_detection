# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/loss.base.ipynb.

# %% auto 0
__all__ = ['BaseLoss']

# %% ../../nbs/loss.base.ipynb 2
from fastai.vision.all import *

# %% ../../nbs/loss.base.ipynb 3
#adopted from simple_faster_rcnn repo
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(gt_label.shape[0],-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(TensorPoint(pred_loc), gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss

# %% ../../nbs/loss.base.ipynb 4
class BaseLoss(Module):
    def __init__(self):
        self.cls_loss_func= CrossEntropyLossFlat(ignore_index=-1)
        self.reg_loss_func= _fast_rcnn_loc_loss
    
    def joint_loss(self, reg_loss, cls_loss):
        loss= (cls_loss+reg_loss)
        return loss.sum()
