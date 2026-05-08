# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

from mmcv.runner import BaseModule, force_fp32
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class SLIP_RS_FCBBoxHead(BaseModule):

    def __init__(self, 
                 roi_feat_size=7,
                 in_channels=256,
                 num_shared_fcs=2,
                 fc_out_channels=1024,
                 norm_cfg=None,
                 init_cfg=None,
                 embed_dim=512,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 *args, 
                 **kwargs):
        super(SLIP_RS_FCBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels  # 256
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.num_shared_fcs = num_shared_fcs
        self.fc_out_channels = fc_out_channels
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        shared_fcs_in_channels = self.in_channels * self.roi_feat_area
        shared_fcs_out_channels = self.fc_out_channels
        self.shared_fcs = \
            self._add_fc_branch(
                num_branch_fcs=self.num_shared_fcs, in_channels=shared_fcs_in_channels, out_channels=shared_fcs_out_channels)

        self.relu = nn.ReLU(inplace=True)

        # reconstruct fc_cls and fc_reg
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        cls_channels = embed_dim

        self.fc_cls = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.fc_out_channels,
            out_features=cls_channels)
        
        out_dim_reg = 4
        self.fc_reg = build_linear_layer(
            self.reg_predictor_cfg,
            in_features=self.fc_out_channels,
            out_features=out_dim_reg)
        
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Normal', std=0.01, override=dict(name='fc_cls')),
                dict(type='Normal', std=0.001, override=dict(name='fc_reg')),
                dict(type='Xavier', distribution='uniform', override=[dict(name='shared_fcs')])]

    def _add_fc_branch(self, num_branch_fcs, in_channels, out_channels):
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                in_channels = (
                    in_channels if i == 0 else out_channels)
                branch_fcs.append(
                    nn.Linear(in_channels, out_channels))
        return branch_fcs

    def forward(self, x, cls_text_embedding_list, rois):
        # shared part
        if self.num_shared_fcs > 0:
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        if x_cls.dim() > 2:
            x_cls = x_cls.flatten(1)

        if x_reg.dim() > 2:
            x_reg = x_reg.flatten(1)

        cls_vis_embeddings = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)

        img_cls_score_list = []
        img_id_mask = rois[:, 0]
        logit_scale = self.logit_scale.exp()
        for img_id in rois[:, 0].unique().tolist():
            img_vis_embedding = cls_vis_embeddings[img_id_mask == img_id]
            img_text_embedding = cls_text_embedding_list[int(img_id)]

            # normalize features and embeddings
            img_vis_embedding = img_vis_embedding / img_vis_embedding.norm(dim=1, keepdim=True)
            img_text_embedding = img_text_embedding / img_text_embedding.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            img_cls_score = logit_scale * img_vis_embedding @ img_text_embedding.t()
            img_cls_score_list.append(img_cls_score)

        return img_cls_score_list, bbox_pred

    def get_targets(self,
                    sampling_results,
                    batch_unique_texts_list,
                    rcnn_train_cfg):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            batch_unique_texts_list,
            cfg=rcnn_train_cfg)

        return labels, label_weights, pos_assigned_gt_inds_list, bbox_targets, bbox_weights
    
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, batch_unique_texts, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples, ),
                                     len(batch_unique_texts) - 1,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            pos_bbox_targets = self.bbox_coder.encode(
                pos_bboxes, pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score_list', 'bbox_pred'))
    def loss(self,
             cls_score_list,
             bbox_pred,
             batch_instance_idx_maps,
             labels,
             label_weights_list,
             pos_assigned_gt_inds_list,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score_list is not None:
            batch = len(cls_score_list)
            cls_loss_list = []
            for img_id in range(batch):
                img_cls_scores = cls_score_list[img_id]
                img_instance_idx_maps = batch_instance_idx_maps[img_id]
                img_labels = labels[img_id]
                img_label_weights = label_weights_list[img_id]
                img_pos_assigned_gt_inds = pos_assigned_gt_inds_list[img_id]

                avg_factor = max(torch.sum(img_label_weights > 0).float().item(), 1.)

                neg_cls_scores = img_cls_scores[img_labels != 0]
                neg_img_labels = img_labels[img_labels != 0]
                neg_positive_map = F.one_hot(neg_img_labels, num_classes=img_cls_scores.shape[-1])
                neg_label_weights = img_label_weights[img_labels != 0]
                neg_loss_cls_ = self.loss_cls(
                    neg_cls_scores,
                    neg_positive_map,
                    neg_label_weights[:, None],
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                
                img_pos_num = len(img_pos_assigned_gt_inds)
                pos_loss_cls_ = 0
                for img_pos_id in range(img_pos_num):
                    pos_assigned_gt_ind = int(img_pos_assigned_gt_inds[img_pos_id])
                    instance_idx_map = img_instance_idx_maps[pos_assigned_gt_ind]
                    pos_instance_cls_score = img_cls_scores[img_pos_id, :][None, :][:, instance_idx_map]
                    pos_instance_label_weight = img_label_weights[None, :][:, img_pos_id]

                    pos_instance_positive_map = F.one_hot(torch.tensor([0]), 
                                                          num_classes=pos_instance_cls_score.shape[-1]).to(pos_instance_cls_score.device)
                    pos_instance_loss_cls_ = self.loss_cls(
                        pos_instance_cls_score,
                        pos_instance_positive_map,
                        pos_instance_label_weight[:, None],
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                    pos_loss_cls_ = pos_loss_cls_ + pos_instance_loss_cls_
                
                cls_loss_list.append(neg_loss_cls_ + pos_loss_cls_)
            losses['loss_cls'] = sum(cls_loss_list) / batch
        if bbox_pred is not None:
            cat_bbox_targets = torch.cat(bbox_targets, dim=0)
            cat_bbox_weights = torch.cat(bbox_weights, dim=0)
            cat_labels = torch.cat(labels, dim=0)
            pos_inds = cat_labels == 0
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    cat_bbox_targets[pos_inds],
                    cat_bbox_weights[pos_inds],
                    avg_factor=cat_bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    
    def forward_test(self, x, cls_text_embedding, rois):
        # shared part
        if self.num_shared_fcs > 0:
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        if x_cls.dim() > 2:
            x_cls = x_cls.flatten(1)

        if x_reg.dim() > 2:
            x_reg = x_reg.flatten(1)

        cls_score = self.fc_cls(x_cls)
        bbox_pred = self.fc_reg(x_reg)

        logit_scale = self.logit_scale.exp()

        # normalize features and embeddings
        cls_score = cls_score / cls_score.norm(dim=1, keepdim=True)
        cls_text_embedding = cls_text_embedding / cls_text_embedding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        cls_score = logit_scale * cls_score @ cls_text_embedding.t()

        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        # some loss (Seesaw loss..) may have custom activation
        scores = cls_score.sigmoid()
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
    
    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)