from .coco import CocoDataset
import os.path as osp
import mmcv
import numpy as np
from mmcv.utils import print_log
from .api_wrappers import COCO
from .api_wrappers import COCOeval
from terminaltables import AsciiTable
import itertools
import logging
from collections import OrderedDict

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class RS_Dataset(CocoDataset):

    CLASSES = None
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 text_template='',
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)

        self.CLASSES = self.get_classes(classes)
        self.text_template = text_template

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx, test_mode=True)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
        
    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
    
    def get_ann_info(self, idx, test_mode=False):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        if test_mode:
            return self._parse_ann_info_test(self.data_infos[idx], ann_info)
        else:
            return self._parse_ann_info(self.data_infos[idx], ann_info)
        
    def _parse_ann_info(self, img_info, ann_info):
        text_prompts = [self.text_template.format(cls_name) for cls_name in self.CLASSES]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        text_list = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                pos_text = text_prompts[self.cat2label[ann['category_id']]]
                new_prompts = [pos_text] + [
                                t for t in text_prompts if t != pos_text
                            ]
                gt_bboxes.append(bbox)
                gt_labels.append(0)
                text_list.append(new_prompts)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            texts=text_list,
            bboxes_ignore=gt_bboxes_ignore)

        return ann
    
    def _parse_ann_info_test(self, img_info, ann_info):
        text_prompts = [self.text_template.format(cls_name) for cls_name in self.CLASSES]
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            texts=text_prompts,
            bboxes_ignore=gt_bboxes_ignore)

        return ann

    def _compute_f1_metrics(self, cocoEval):
        """Compute dataset-level precision/recall/F1 from COCO matches."""
        params = cocoEval.params
        cat_ids = params.catIds if params.useCats else [-1]
        img_ids = params.imgIds
        area_labels = getattr(params, 'areaRngLbl', None)
        num_areas = len(params.areaRng)
        num_imgs = len(img_ids)
        iou_thrs = params.iouThrs

        if area_labels is not None and 'all' in area_labels:
            area_idx = area_labels.index('all')
        else:
            area_idx = 0

        tp = np.zeros(len(iou_thrs), dtype=np.float64)
        fp = np.zeros(len(iou_thrs), dtype=np.float64)
        fn = np.zeros(len(iou_thrs), dtype=np.float64)

        for cat_idx in range(len(cat_ids)):
            base_idx = cat_idx * num_areas * num_imgs + area_idx * num_imgs
            for img_offset in range(num_imgs):
                eval_img = cocoEval.evalImgs[base_idx + img_offset]
                if eval_img is None:
                    continue

                dt_matches = eval_img['dtMatches']
                gt_matches = eval_img['gtMatches']
                dt_ignore = eval_img['dtIgnore'].astype(bool)
                gt_ignore = eval_img['gtIgnore'].astype(bool)

                tp += np.sum((dt_matches > 0) & (~dt_ignore), axis=1)
                fp += np.sum((dt_matches == 0) & (~dt_ignore), axis=1)
                fn += np.sum((gt_matches == 0) & (~gt_ignore[None, :]), axis=1)

        precision = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0)

        iou_to_index = {
            round(float(iou_thr), 2): idx for idx, iou_thr in enumerate(iou_thrs)
        }
        iou50_idx = iou_to_index.get(0.50, 0)

        return OrderedDict([
            ('iou_thrs', iou_thrs),
            ('precision', precision),
            ('recall', recall),
            ('f1', f1),
            ('precision_50', float(precision[iou50_idx])),
            ('recall_50', float(recall[iou50_idx])),
            ('f1_50', float(f1[iou50_idx])),
            ('precision_miou', float(np.mean(precision))),
            ('recall_miou', float(np.mean(recall))),
            ('f1_miou', float(np.mean(f1))),
        ])

    def _filter_results_by_gt_categories(self, results):
        """Keep only predictions whose categories appear in image GT."""
        filtered_results = []
        for idx, result in enumerate(results):
            ann_info = self.get_ann_info(idx, test_mode=True)
            valid_label_set = set(ann_info['labels'].tolist())

            if isinstance(result, tuple):
                det, seg = result
                filtered_det = []
                filtered_seg = []
                filtered_scores = [] if isinstance(seg, tuple) else None

                for label, bboxes in enumerate(det):
                    if label in valid_label_set:
                        filtered_det.append(bboxes.copy())
                        if isinstance(seg, tuple):
                            filtered_seg.append(list(seg[0][label]))
                            filtered_scores.append(list(seg[1][label]))
                        else:
                            filtered_seg.append(list(seg[label]))
                    else:
                        filtered_det.append(np.zeros((0, 5), dtype=bboxes.dtype))
                        if isinstance(seg, tuple):
                            filtered_seg.append([])
                            filtered_scores.append([])
                        else:
                            filtered_seg.append([])

                if isinstance(seg, tuple):
                    filtered_results.append((filtered_det,
                                             (filtered_seg, filtered_scores)))
                else:
                    filtered_results.append((filtered_det, filtered_seg))
            else:
                filtered_result = []
                for label, bboxes in enumerate(result):
                    if label in valid_label_set:
                        filtered_result.append(bboxes.copy())
                    else:
                        filtered_result.append(np.zeros((0, 5), dtype=bboxes.dtype))
                filtered_results.append(filtered_result)

        return filtered_results

    def _build_coco_dt_from_results(self, results, metric):
        if metric == 'segm':
            json_results = self._segm2json(results)[1]
        else:
            json_results = self._det2json(results)
        return self.coco.loadRes(json_results)

    def _log_f1_metrics(self, metric, f1_metrics, logger=None, title=None):
        title = title or f'{metric} Precision/Recall/F1 summary'
        table_data = [['metric', 'IoU=0.50', 'IoU=0.50:0.95']]
        table_data.append([
            'Precision',
            f'{f1_metrics["precision_50"]:.4f}',
            f'{f1_metrics["precision_miou"]:.4f}'
        ])
        table_data.append([
            'Recall',
            f'{f1_metrics["recall_50"]:.4f}',
            f'{f1_metrics["recall_miou"]:.4f}'
        ])
        table_data.append([
            'F1',
            f'{f1_metrics["f1_50"]:.4f}',
            f'{f1_metrics["f1_miou"]:.4f}'
        ])
        table = AsciiTable(table_data)
        print_log(f'\n{title}\n{table.table}',
                  logger=logger)
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=True,
                 proposal_nums=(100, 300, 1500),
                 iou_thrs=None,
                 metric_items=None):

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_vt': 3,
                'mAP_t': 4,
                'mAP_s': 5,
                'mAP_m': 6,
                'AR@100': 7,
                'AR@300': 8,
                'AR@1500': 9,
                'AR_vt@1500': 10,
                'AR_t@1500': 11,
                'AR_s@1500': 12,
                'AR_m@1500': 13
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                f1_metrics = self._compute_f1_metrics(cocoEval)
                self._log_f1_metrics(metric, f1_metrics, logger=logger)

                gt_filtered_results = self._filter_results_by_gt_categories(results)
                cocoDt_gt_filtered = self._build_coco_dt_from_results(
                    gt_filtered_results, metric)
                cocoEval_gt_filtered = COCOeval(cocoGt, cocoDt_gt_filtered,
                                                iou_type)
                cocoEval_gt_filtered.params.catIds = self.cat_ids
                cocoEval_gt_filtered.params.imgIds = self.img_ids
                cocoEval_gt_filtered.params.maxDets = list(proposal_nums)
                cocoEval_gt_filtered.params.iouThrs = iou_thrs
                cocoEval_gt_filtered.evaluate()
                cocoEval_gt_filtered.accumulate()
                gt_filtered_f1_metrics = self._compute_f1_metrics(
                    cocoEval_gt_filtered)
                self._log_f1_metrics(
                    metric,
                    gt_filtered_f1_metrics,
                    logger=logger,
                    title=(f'{metric} Precision/Recall/F1 summary '
                           f'(predictions limited to image GT classes)'))

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_vt', 'mAP_t', 'mAP_s', 'mAP_m'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    if metric_item == 'mAP':
                        AP_cls_list = [float(AP_cls) for _, AP_cls in results_per_category]
                        mAP = sum(AP_cls_list) / len(AP_cls_list)
                        val = float(mAP)
                    else:
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                        )
                    eval_results[key] = val
                eval_results[f'{metric}_precision_50'] = f1_metrics['precision_50']
                eval_results[f'{metric}_recall_50'] = f1_metrics['recall_50']
                eval_results[f'{metric}_f1_50'] = f1_metrics['f1_50']
                eval_results[f'{metric}_precision_miou'] = f1_metrics['precision_miou']
                eval_results[f'{metric}_recall_miou'] = f1_metrics['recall_miou']
                eval_results[f'{metric}_f1_miou'] = f1_metrics['f1_miou']
                eval_results[f'{metric}_gtcls_precision_50'] = (
                    gt_filtered_f1_metrics['precision_50'])
                eval_results[f'{metric}_gtcls_recall_50'] = (
                    gt_filtered_f1_metrics['recall_50'])
                eval_results[f'{metric}_gtcls_f1_50'] = (
                    gt_filtered_f1_metrics['f1_50'])
                eval_results[f'{metric}_gtcls_precision_miou'] = (
                    gt_filtered_f1_metrics['precision_miou'])
                eval_results[f'{metric}_gtcls_recall_miou'] = (
                    gt_filtered_f1_metrics['recall_miou'])
                eval_results[f'{metric}_gtcls_f1_miou'] = (
                    gt_filtered_f1_metrics['f1_miou'])
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
