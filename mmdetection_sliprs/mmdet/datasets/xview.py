from .builder import DATASETS
from .coco import CocoDataset

import itertools
import logging
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from .api_wrappers import COCOeval
from terminaltables import AsciiTable

from .builder import DATASETS


@DATASETS.register_module()
class XviewDataset(CocoDataset):

    CLASSES = ('Fixed-wing Aircraft',  'Small Aircraft',  'Cargo Plane',  'Helicopter',  
         'Passenger Vehicle',  'Small Car',  'Bus',  'Pickup Truck',  
         'Utility Truck',  'Truck',  'Cargo Truck',  'Truck w/Box',  
         'Truck Tractor',  'Trailer',  'Truck w/Flatbed',  'Truck w/Liquid',  
         'Crane Truck',  'Railway Vehicle',  'Passenger Car',  'Cargo Car',  
         'Flat Car',  'Tank car',  'Locomotive',  'Maritime Vessel',  
         'Motorboat',  'Sailboat',  'Tugboat',  'Barge',  
         'Fishing Vessel',  'Ferry',  'Yacht',  'Container Ship',  
         'Oil Tanker',  'Engineering Vehicle',  'Tower crane',  'Container Crane',  
         'Reach Stacker',  'Straddle Carrier',  'Mobile Crane',  'Dump Truck',  
         'Haul Truck',  'Scraper/Tractor',  'Front loader/Bulldozer',  'Excavator',  
         'Cement Mixer',  'Ground Grader',  'Hut/Tent',  'Shed',  
         'Building',  'Aircraft Hangar',  'Damaged Building',  'Facility',  
         'Construction Site',  'Vehicle Lot',  'Helipad',  'Storage Tank',  
         'Shipping container lot',  'Shipping Container',  'Pylon',  'Tower')

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
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results