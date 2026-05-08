import torch
import CLIP.clip as clip
from collections import OrderedDict
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_text_encoder


@DETECTORS.register_module()
class SLIP_RS(BaseDetector):
    """Implementation of SLIP-RS"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 text_encoder=None,
                 text_pad_length=None,
                 pretrained=None,
                 init_cfg=None):
        super(SLIP_RS, self).__init__(init_cfg)
        self.text_pad_length = text_pad_length

        # cfg for backbone
        if pretrained:
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        # cfg for text_encoder
        self.text_encoder = build_text_encoder(text_encoder)
        text_total_params = sum(p.numel() for p in self.text_encoder.parameters())
        text_trainable_params = sum(
            p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        print('Text encoder params: '
              f'trainable={text_trainable_params:,} / total={text_total_params:,}')

        # cfg for neck and rpn head
        self.neck = build_neck(neck)
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_ = rpn_head.copy()
        rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.rpn_head = build_head(rpn_head_)

        # cfg for rcnn head
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def extract_text_embedding(self, batch_text_prompts, device=None, train_mode=True):
        """Directly extract text-embeddings from the text-encoder."""
        if train_mode:
            batch_unique_texts = []
            batch_instance_idx_maps = []

            for img_prompts in batch_text_prompts:
                img_text_to_idx = OrderedDict()
                instance_idx_maps = []
                instance_gt_labels = []

                for instance_prompts in img_prompts:
                    instance_indices = []
                    for txt in instance_prompts:
                        if txt not in img_text_to_idx:
                            img_text_to_idx[txt] = len(img_text_to_idx)
                        instance_indices.append(img_text_to_idx[txt])
                    instance_idx_maps.append(instance_indices)
                    instance_gt_labels.append(instance_indices[0])

                img_unique_texts = list(img_text_to_idx.keys())
                img_unique_texts.append("background")

                img_text_embeddings = clip.tokenize(img_unique_texts).to(device)
                img_text_embeddings = self.text_encoder(img_text_embeddings) * self.text_encoder.logit_scale / self.text_encoder.logit_scale
                batch_unique_texts.append(img_text_embeddings)
                batch_instance_idx_maps.append(instance_idx_maps)

                instance_gt_labels = torch.tensor(instance_gt_labels, dtype=torch.long, device=device)

            return batch_unique_texts, batch_instance_idx_maps
        else:
            # get the text embeddings
            prompts = batch_text_prompts + ["background"]
            with torch.no_grad():
                img_text_embeddings = clip.tokenize(prompts).to(device)
                img_text_embeddings = self.text_encoder(img_text_embeddings)

            return img_text_embeddings

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      text_prompts,
                      gt_bboxes_ignore=None,
                      **kwargs):

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg,
            **kwargs)
        losses.update(rpn_losses)

        roi_losses = self.roi_head_loss(x, img_metas, proposal_list, gt_bboxes, 
                                        gt_labels, text_prompts, gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses
    
    def roi_head_loss(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, 
                      text_prompts, gt_bboxes_ignore, **kwargs):
        
        batch_unique_texts_list, batch_instance_idx_maps = \
                            self.extract_text_embedding(text_prompts, x[0].device)
        
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, batch_unique_texts_list, 
                                                 batch_instance_idx_maps, gt_bboxes_ignore, **kwargs)
        
        return roi_losses
    
    def forward_test(self, imgs, img_metas, **kwargs):

        return self.simple_test(imgs[0], img_metas[0], **kwargs)
        
    def simple_test(self, img, img_metas, rescale=False, text_prompts=None, **kwargs):
        
        assert text_prompts != None, 'text_prompts must be implemented.'
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)

        text_embeddings = self.extract_text_embedding(text_prompts[0][0], device=x[0].device, train_mode=False)

        return self.roi_head.simple_test(
            x, proposal_list, text_embeddings, img_metas, rescale=rescale)
    
    def aug_test(self, imgs, img_metas, rescale=False):

        NotImplementedError
    
    
