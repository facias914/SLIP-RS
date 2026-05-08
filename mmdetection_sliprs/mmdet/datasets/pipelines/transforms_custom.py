import copy
import random

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES
from .transforms import Mosaic, RandomAffine, RandomCrop, find_inside_bboxes


@PIPELINES.register_module()
class MosaicCustom(Mosaic):
    """Mosaic that keeps text prompts aligned with mixed gt boxes."""

    def __init__(self, *args, pre_crop_size=None, crop_attempts=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_crop_size = pre_crop_size
        self.crop_attempts = crop_attempts

    def _crop_patch(self, results_patch):
        if self.pre_crop_size is None:
            return results_patch

        crop_h, crop_w = self.pre_crop_size
        img = results_patch['img']
        img_h, img_w = img.shape[:2]
        crop_h = min(crop_h, img_h)
        crop_w = min(crop_w, img_w)

        for _ in range(self.crop_attempts):
            margin_h = max(img_h - crop_h, 0)
            margin_w = max(img_w - crop_w, 0)
            offset_h = random.randint(0, margin_h) if margin_h > 0 else 0
            offset_w = random.randint(0, margin_w) if margin_w > 0 else 0
            crop_y1, crop_y2 = offset_h, offset_h + crop_h
            crop_x1, crop_x2 = offset_w, offset_w + crop_w

            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results_patch['gt_bboxes'] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, cropped_img.shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, cropped_img.shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])

            if valid_inds.any():
                results_patch['img'] = cropped_img
                results_patch['img_shape'] = cropped_img.shape
                results_patch['gt_bboxes'] = bboxes[valid_inds, :]
                results_patch['gt_labels'] = results_patch['gt_labels'][valid_inds]
                if 'text_prompts' in results_patch:
                    keep_inds = np.nonzero(valid_inds)[0].tolist()
                    results_patch['text_prompts'] = [
                        results_patch['text_prompts'][i] for i in keep_inds
                    ]
                return results_patch

        return results_patch

    def _mosaic_transform(self, results):
        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        mosaic_text_prompts = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            results_patch = self._crop_patch(results_patch)

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']
            text_prompts_i = results_patch.get('text_prompts', None)

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
            if text_prompts_i is not None:
                mosaic_text_prompts.extend(text_prompts_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                bbox_w = mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0]
                bbox_h = mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1]
                valid_inds = np.nonzero((bbox_w > self.min_bbox_size) &
                                        (bbox_h > self.min_bbox_size))[0]
                mosaic_bboxes = mosaic_bboxes[valid_inds]
                mosaic_labels = mosaic_labels[valid_inds]
                if mosaic_text_prompts:
                    mosaic_text_prompts = [
                        mosaic_text_prompts[i] for i in valid_inds.tolist()
                    ]
        else:
            mosaic_bboxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.array([], dtype=np.int64)

        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        if mosaic_text_prompts:
            keep_inds = np.nonzero(inside_inds)[0].tolist()
            mosaic_text_prompts = [mosaic_text_prompts[i] for i in keep_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels
        if 'text_prompts' in results:
            results['text_prompts'] = mosaic_text_prompts

        return results

    def get_indexes(self, dataset):
        return [random.randint(0, len(dataset) - 1) for _ in range(3)]


@PIPELINES.register_module()
class RandomAffineCustom(RandomAffine):
    """RandomAffine that keeps text prompts aligned with filtered gt boxes."""

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                warp_bboxes = np.vstack(
                    (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = \
                        warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = \
                        warp_bboxes[:, [1, 3]].clip(0, height)

                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    filter_index = self.filter_gt_bboxes(
                        bboxes * scaling_ratio, warp_bboxes)
                    valid_index = valid_index & filter_index

                results[key] = warp_bboxes[valid_index]
                if key in ['gt_bboxes']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]
                    if 'text_prompts' in results:
                        keep_inds = np.nonzero(valid_index)[0].tolist()
                        results['text_prompts'] = [
                            results['text_prompts'][i] for i in keep_inds
                        ]

                if 'gt_masks' in results:
                    raise NotImplementedError(
                        'RandomAffineCustom only supports bbox.')
        return results


@PIPELINES.register_module()
class RandomCropCustom(RandomCrop):
    """RandomCrop that keeps text prompts aligned with cropped gt boxes."""

    def _crop_data(self, results, crop_size, allow_negative_crop):
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]

            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            if key == 'gt_bboxes' and 'text_prompts' in results:
                keep_inds = np.nonzero(valid_inds)[0].tolist()
                results['text_prompts'] = [
                    results['text_prompts'][i] for i in keep_inds
                ]

            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        return results
