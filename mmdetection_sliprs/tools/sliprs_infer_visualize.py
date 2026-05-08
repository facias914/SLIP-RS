#!/usr/bin/env python
import argparse
import ast
import os
import os.path as osp
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.apis import init_detector  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402
from mmdet.datasets.pipelines import Compose  # noqa: E402


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='SLIP-RS image inference and visualization.')
    parser.add_argument('config', help='model config file')
    parser.add_argument('checkpoint', help='model checkpoint file')
    parser.add_argument('image', help='image file or image directory')
    parser.add_argument(
        '--prompt',
        required=True,
        nargs='+',
        help=('text prompt(s). Examples: --prompt "small-vehicle", or --prompt [plane, ship]'))
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold for visualization')
    parser.add_argument(
        '--out-dir',
        default='tools/sliprs_vis_results',
        help='directory to save visualized images')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='device used for inference, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--palette',
        default='random',
        help='bbox color palette used by MMDetection visualization')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show visualization window while saving results')
    return parser.parse_args()


def normalize_prompts(prompt_args):
    if len(prompt_args) == 1:
        raw = prompt_args[0]
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, str):
            prompts = [parsed]
        elif isinstance(parsed, (list, tuple)):
            prompts = [str(item) for item in parsed]
        else:
            prompts = [item.strip() for item in raw.split(',') if item.strip()]
    else:
        prompts = prompt_args

    prompts = [str(item).strip() for item in prompts if str(item).strip()]
    if not prompts:
        raise ValueError('At least one non-empty prompt is required.')
    return prompts


def collect_images(image):
    if isinstance(image, np.ndarray):
        return [image]

    image = osp.abspath(osp.expanduser(image))
    if osp.isdir(image):
        files = []
        for name in sorted(os.listdir(image)):
            path = osp.join(image, name)
            if osp.isfile(path) and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                files.append(path)
        if not files:
            raise FileNotFoundError(f'No image files found in {image}')
        return files

    if not osp.isfile(image):
        raise FileNotFoundError(f'Image path does not exist: {image}')
    return [image]


def build_empty_ann(prompts):
    return dict(
        bboxes=np.zeros((0, 4), dtype=np.float32),
        labels=np.array([], dtype=np.int64),
        bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
        texts=list(prompts))


def get_test_pipeline_cfg(cfg):
    test_cfg = cfg.data.test
    if 'pipeline' in test_cfg:
        return test_cfg.pipeline
    if 'datasets' in test_cfg and len(test_cfg.datasets) > 0:
        return test_cfg.datasets[0].pipeline
    if isinstance(test_cfg, (list, tuple)) and len(test_cfg) > 0:
        return test_cfg[0].pipeline
    raise AttributeError('Cannot find test pipeline in cfg.data.test.')


def inference_sliprs(model, image, prompts):
    cfg = model.cfg.copy()
    pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(image, np.ndarray):
        pipeline = pipeline.copy()
        pipeline[0].type = 'LoadImageFromWebcam'

    pipeline = replace_ImageToTensor(pipeline)
    test_pipeline = Compose(pipeline)

    if isinstance(image, np.ndarray):
        data = dict(img=image)
    else:
        data = dict(img_info=dict(filename=image), img_prefix=None)

    data.update(ann_info=build_empty_ann(prompts), bbox_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if 'text_prompts' in data and hasattr(data['text_prompts'], 'data'):
        data['text_prompts'] = data['text_prompts'].data

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    with torch.no_grad():
        return model(return_loss=False, rescale=True, **data)[0]


def output_path_for(image, out_dir, index):
    if isinstance(image, np.ndarray):
        return osp.join(out_dir, f'image_{index:06d}.jpg')
    return osp.join(out_dir, Path(image).stem + '_vis' + Path(image).suffix)


def main():
    args = parse_args()
    prompts = normalize_prompts(args.prompt)

    config = osp.abspath(osp.expanduser(args.config))
    checkpoint = osp.abspath(osp.expanduser(args.checkpoint))
    image_input = args.image
    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    mmcv.mkdir_or_exist(out_dir)

    os.chdir(str(REPO_ROOT))
    model = init_detector(config, checkpoint, device=args.device)
    model.CLASSES = tuple(prompts) + ('background',)

    images = collect_images(image_input)
    for idx, image in enumerate(images):
        result = inference_sliprs(model, image, prompts)
        out_file = output_path_for(image, out_dir, idx)
        model.show_result(
            image,
            result,
            score_thr=args.score_thr,
            bbox_color=args.palette,
            text_color=args.palette,
            show=args.show,
            out_file=out_file)
        print(f'Saved visualization: {out_file}')


if __name__ == '__main__':
    main()
