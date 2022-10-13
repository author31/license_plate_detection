import cv2
import numpy as np
from fastai.vision.all import *

def draw_bboxes(img, bbox, identities=None, offset=(0,0), score_thr=0.5):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        color = (0, 255, 10)
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
    return img 

def calc_area(bbox):
    return np.maximum(bbox[:, :,2]-bbox[:, :, 0], 0)*\
           np.maximum(bbox[:, :, 3]-bbox[:, :, 1], 0)


def calc_pairwise_iou(b1, b2):
    b1_inters=np.maximum(b1[None, :, :2], b2[:, None, :2])
    b2_inters=np.minimum(b1[None, :, 2:], b2[:, None, 2:])
    inters_coor = np.concatenate((b1_inters, b2_inters), axis=-1)
    inters_area= calc_area(inters_coor)
    area_1=calc_area(b1[:, None, :])
    area_2=calc_area(b2[:, None, :])
    union=area_1[None, :, :]+area_2[:, None, :]
    iou = inters_area / union.squeeze()
    return iou

def calc_iou(b1, b2):
    b1_inters=np.maximum(b1[:, :2], b2[:, :2])
    b2_inters=np.minimum(b1[:, 2:], b2[:, 2:])
    inters_coor = np.concatenate((b1_inters, b2_inters), axis=1)
    inters_area= calc_area(inters_coor)
    area_1=calc_area(b1)
    area_2=calc_area(b2)
    union=area_1+area_2-inters_area
    iou = inters_area / union
    return iou

def xyxy2xywh(bbox):
    w= bbox[:, 2]- bbox[:, 0]
    h= bbox[:, 3]- bbox[:, 1]
    x= bbox[:, 0]+ (w/2)
    y= bbox[:, 1]+ (h/2)
    return np.vstack([x,y,w,h]).T

def xywh2xyxy(bbox):
    x1= bbox[:, 0]- (bbox[:, 2]/2)
    y1= bbox[:, 1]- (bbox[:, 3]/2)
    x2= bbox[:, 0]+ (bbox[:, 2]/2)
    y2= bbox[:, 1]+ (bbox[:, 3]/2)
    return np.vstack([x1,y1,x2,y2]).T

def calc_offset(gt_bboxes, anchors):
    cvt_bbox= xyxy2xywh(gt_bboxes)
    cvt_anchors= xyxy2xywh(anchors)
    x_del= (cvt_bbox[:, 0]-cvt_anchors[:, 0])/cvt_anchors[:, 2]
    y_del= (cvt_bbox[:, 1]-cvt_anchors[:, 1])/cvt_anchors[:, 3]
    w_del= np.log((cvt_bbox[:, 2])/(cvt_anchors[:, 2]))
    h_del= np.log((cvt_bbox[:, 3])/(cvt_anchors[:, 3]))
    return np.stack([x_del, y_del, w_del, h_del]).T

def reverse_offset(anchors, offsets):
    cvt_anchors= xyxy2xywh(anchors)
    x= offsets[:, 0]*cvt_anchors[:, 2]+cvt_anchors[:, 0]
    y= offsets[:, 1]*cvt_anchors[:, 3]+cvt_anchors[:, 1]
    w= np.exp(offsets[:, 2])*cvt_anchors[:, 2]
    h= np.exp(offsets[:, 3])*cvt_anchors[:, 3]
    bbox=np.stack([x,y,w,h]).T
    bbox=xywh2xyxy(bbox)
    return bbox

def sample_anchor(anchors, flags, num_per_img=128, pos_fraction=0.5):
    anchor_labels= np.full((anchors.shape[0],),-1)
    fg_ind= np.where(flags==1)[0]
    bg_ind= np.where(flags==0)[0]
    if fg_ind.shape[0] > (fg_lim:=num_per_img*pos_fraction):
        fg_ind= np.random.choice(fg_ind, int(fg_lim), replace=False)
    bg_lim= num_per_img-fg_ind.shape[0]
    bg_ind= np.random.choice(bg_ind, bg_lim, replace=False)
    anchor_labels[fg_ind]=1
    anchor_labels[bg_ind]=0
    return anchor_labels

def assign_anchor(all_anchors, gt_bboxes, pos_iou_thr=0.7, neg_iou_thr=0.3):
    pair_ious= calc_pairwise_iou(all_anchors, gt_bboxes)
    flags= np.full(all_anchors.shape[0], fill_value=-1)
    for idx, _ in enumerate(pair_ious):
        valid_ind= np.where(flags==-1)
        iou= pair_ious[idx, valid_ind].squeeze()
        pos_ind= np.where((iou>pos_iou_thr)|
                         (iou==iou.max()))
        flags[pos_ind]=1
    neg= pair_ious.max(axis=0)<0.3
    neg_ind= np.where((neg==True)&(flags==-1))
    flags[neg_ind]= 0
    return flags

def unscale_pnts(y, sz): 
    return TensorPoint((y+1) * tensor(sz).float()/2, img_size=sz)
