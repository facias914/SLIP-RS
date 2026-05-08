import numpy as np
from mmcv.parallel import DataContainer as DC
from .formatting import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadAnnotationsCustom:

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_text=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_text = with_text
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results
    
    def _load_texts(self, results):

        results['text_prompts'] = results['ann_info']['texts'].copy()
        return results

    def __call__(self, results):

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_text:
            results = self._load_texts(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_text={self.with_text}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str
    

@PIPELINES.register_module()
class DefaultFormatBundleCustom:

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            results['img'] = DC(
                img, padding_value=self.pad_val['img'], stack=True)
        for key in ['gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'text_prompts' in results:
            results['text_prompts'] = DC(results['text_prompts'], cpu_only=True)
        return results

    def _add_default_meta_keys(self, results):
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'