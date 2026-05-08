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
import random
from collections import OrderedDict, defaultdict
import copy
from .builder import DATASETS
from .pipelines import Compose


class AttributeContrastiveBuilder:
    def __init__(self, attr_dict, drop_prob=0.5, shuffle_prob=1.0, max_negatives=10, use_value=True):
        self.attr_dict = attr_dict
        self.drop_prob = drop_prob
        self.shuffle_prob = shuffle_prob
        self.max_negatives = max_negatives
        self.use_value = use_value

    def build_positive_text(self, category_name, ann_attributes):
        ProtoTag = ann_attributes.pop("ProtoTag", None)
        original_attrs = [(k, v) for k, v in ann_attributes.items()]

        # random drop
        attrs = [(k, v) for k, v in original_attrs if random.random() > self.drop_prob]
        if len(attrs) == 0:
            attrs = [random.choice(original_attrs)]

        # random shuffle
        if random.random() < self.shuffle_prob:
            random.shuffle(attrs)

        if self.use_value:
            parts = [f"{v}" for _, v in attrs]
        else:
            parts = [f"{k}" for k, _ in attrs]

        text = f"{category_name} + " + " + ".join(parts) if parts else category_name
        return text, attrs

    def build_negative_text(self, proto, attrs):
        candidates_per_attr = []
        for k, v in attrs:
            all_vals = self.attr_dict.get(proto, {}).get(k, [])
            if not all_vals:
                all_vals = [v]
            candidates_per_attr.append(all_vals)

        all_combinations = list(itertools.product(*candidates_per_attr))

        positive_values = tuple(v for _, v in attrs)
        neg_texts = []

        for combo in all_combinations:
            if combo == positive_values:
                continue
            parts = [val if self.use_value else key for val, (key, _) in zip(combo, attrs)]
            neg_texts.append(f"{proto} + " + " + ".join(parts))

        if len(neg_texts) > self.max_negatives:
            neg_texts = random.sample(neg_texts, self.max_negatives)

        return neg_texts

    def __call__(self, category_name, ann_attributes=None, prototag_list=None):
        if ann_attributes is not None:
            pos_text, choosed_attrs = self.build_positive_text(category_name, ann_attributes)
            neg_text = self.build_negative_text(category_name, choosed_attrs)
        elif prototag_list is not None:
            pos_text = category_name
            neg_text = [x for x in prototag_list if x != pos_text]
        return pos_text, neg_text
    

def rebuild_attribute_categories(coco, test_attri):
    print(f"\nRebuilding categories based on attribute: {test_attri}")

    original_category = list(coco.cats.values())[0]["name"]

    attr_values = set()
    for ann in coco.dataset["annotations"]:
        if test_attri in ann["attributes"]:
            attr_values.add(ann["attributes"][test_attri])

    attr_values = sorted(list(attr_values))
    print(f"Found {len(attr_values)} attribute values for '{test_attri}':")
    print(attr_values)

    new_categories = []
    attr_value_to_new_id = {}

    for idx, val in enumerate(attr_values):
        new_cat = {
            "id": idx,
            "name": f"{original_category} + {val}",
            "supercategory": original_category
        }
        new_categories.append(new_cat)
        attr_value_to_new_id[val] = idx

    print("\nNew categories:")
    for c in new_categories:
        print(c["id"], c["name"])

    new_anns = {}
    new_imgToAnns = defaultdict(list)
    new_catToImgs = defaultdict(list)

    for ann_id, ann in coco.anns.items():
        ann = copy.deepcopy(ann)

        if test_attri in ann["attributes"]:
            val = ann["attributes"][test_attri]
            ann["category_id"] = attr_value_to_new_id[val]
        else:
            ann["category_id"] = -1

        new_anns[ann_id] = ann
        new_imgToAnns[ann["image_id"]].append(ann)
        new_catToImgs[ann["category_id"]].append(ann["image_id"])

    new_dataset = copy.deepcopy(coco.dataset)
    new_dataset["categories"] = new_categories
    new_dataset["annotations"] = list(new_anns.values())

    coco.anns = new_anns
    coco.imgToAnns = new_imgToAnns
    coco.catToImgs = new_catToImgs
    coco.cats = {c["id"]: c for c in new_categories}
    coco.dataset = new_dataset 

    print("\nRebuild COMPLETED.")
    print(f"Total new categories: {len(new_categories)}")

    return coco


@DATASETS.register_module()
class RS_Dataset_Attri(CocoDataset):
    
    def __init__(self,
                 ann_file,
                 pipeline,
                 attri_dict=None,
                 data_root=None,
                 img_prefix='',
                 test_mode=False,
                 test_cls=None,
                 test_attri=None,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient(**file_client_args)

        if test_mode:
            test_attris = attri_dict[test_cls][test_attri]
            self.CLASSES = [f"{test_cls} + {item}" for item in test_attris]
        else:
            self.CLASSES = list(attri_dict.keys())
            self.text_builder = AttributeContrastiveBuilder(attri_dict)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)

        # load annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                if test_mode:
                    self.data_infos = self.load_annotations_test(local_path, test_attri)
                else:
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
    
    def load_annotations_test(self, ann_file, test_attri):
        self.coco = COCO(ann_file)
        self.coco = rebuild_attribute_categories(self.coco, test_attri)
        self.CLASSES = [item['name'] for item in self.coco.dataset["categories"]]
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
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        text_list = []
        for i, ann in enumerate(ann_info):
            category_name = self.CLASSES[self.cat2label[ann['category_id']]]
            ann_attributes = ann['attributes']
            if len(ann_attributes) == 1 and 'ProtoTag' in ann_attributes.keys():
                pos_text, neg_text = self.text_builder(category_name, prototag_list=self.CLASSES)
            else:
                pos_text, neg_text = self.text_builder(category_name, ann_attributes=ann_attributes)
            neg_text[:0] = [pos_text]
            text_list.append(neg_text)
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
                gt_labels.append(0)

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
        text_prompts = self.CLASSES
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
