"""
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import torchvision

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


class ToTensor(object):
    def __call__(self, img, target, depth=None):
        return (F.to_tensor(img), target, F.to_tensor(depth)) if depth is not None else (F.to_tensor(img), target)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None, depth=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return (image, None, depth) if depth is not None else (image, None)
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return (image, target, depth) if depth is not None else (image, target)
    

def resize(image, target, size, max_size=None, depth=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]  # (h, w)
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)

    # resize img
    rescaled_image = F.resize(image, size)

    # resize depth
    rescaled_depth = None
    if depth is not None:
        rescaled_depth = F.resize(depth, size, interpolation=Image.NEAREST)

    if target is None:
        return (rescaled_image, None) if rescaled_depth is None else (rescaled_image, None, rescaled_depth)

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest"
        )[:, 0] > 0.5

    return (rescaled_image, target) if rescaled_depth is None else (rescaled_image, target, rescaled_depth)

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, depth=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, depth=depth)

    
class RandomHorizontalFlip(object):
    "only for train"
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, target, depth=None):
        if random.random() < self.prob:
            img = F.hflip(img)
            if depth is not None:
                depth = F.hflip(depth)

            w, _ = img.size
            target = target.copy()
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
                target["boxes"] = boxes

            if "masks" in target:
                target['masks'] = target['masks'].flip(-1)

        return (img, target, depth) if depth is not None else (img, target)

    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target, depth=None):
        for t in self.transforms:
            if depth is not None:
                img, target, depth = t(img, target, depth)
            else:
                img, target = t(img, target)
        return (img, target, depth) if depth is not None else (img, target)
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string



def horiz_flip(img, bbox, gazex, gazey, inout, depth=None):
    width, height = img.size
    img = torchvision.transforms.functional.hflip(img)
    if depth:
        depth = torchvision.transforms.functional.hflip(depth)
    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]
    if inout:
        gazex = [width - x for x in gazex]

    return img, bbox, gazex, gazey, depth


def random_bbox_jitter(img, bbox):
    width, height = img.size
    xmin, ymin, xmax, ymax = bbox
    jitter = 0.2
    xmin_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    xmax_j = (np.random.random_sample() * (jitter*2) - jitter) * (xmax - xmin)
    ymin_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)
    ymax_j = (np.random.random_sample() * (jitter*2) - jitter) * (ymax - ymin)

    bbox = [max(0, xmin_j + xmin), max(0, ymin_j + ymin), min(width, xmax_j + xmax), min(height, ymax_j + ymax)]

    return bbox


def random_crop(img, bbox, gazex, gazey, inout, depth=None):
    width, height = img.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    
    # determine feasible crop region (must include bbox and gaze target)
    crop_reg_xmin = min(bbox_xmin, min(gazex)) if inout else bbox_xmin
    crop_reg_ymin = min(bbox_ymin, min(gazey)) if inout else bbox_ymin
    crop_reg_xmax = max(bbox_xmax, max(gazex)) if inout else bbox_xmax
    crop_reg_ymax = max(bbox_ymax, max(gazey)) if inout else bbox_ymax

    # Ensure crop region is within image boundaries，（后面可以对json文件进行处理，对json文件中略微超出的进行规范）
    crop_reg_xmin = max(0, crop_reg_xmin)
    crop_reg_ymin = max(0, crop_reg_ymin)
    crop_reg_xmax = min(width, crop_reg_xmax)
    crop_reg_ymax = min(height, crop_reg_ymax)
    
    try:
        xmin = random.randint(0, int(crop_reg_xmin))
        ymin = random.randint(0, int(crop_reg_ymin))
        xmax = random.randint(int(crop_reg_xmax), width)
        ymax = random.randint(int(crop_reg_ymax), height)
    except Exception as e:
        print(f"img size: {width}x{height}")
        print(f"bbox: {bbox_xmin} {bbox_ymin} {bbox_xmax} {bbox_ymax}")
        print(f"gazex: {gazex},gazey: {gazey}, inout: {inout}")
        print(f"crop region: {crop_reg_xmin} {crop_reg_ymin} {crop_reg_xmax} {crop_reg_ymax}")
        raise ValueError(f"Invalid crop region: {crop_reg_xmin}, {crop_reg_ymin}, {crop_reg_xmax}, {crop_reg_ymax}") from e

    img = torchvision.transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    if depth:
        depth = torchvision.transforms.functional.crop(depth, ymin, xmin, ymax - ymin, xmax - xmin)
    bbox = [bbox_xmin - xmin, bbox_ymin - ymin, bbox_xmax - xmin, bbox_ymax - ymin]
    gazex = [x - xmin for x in gazex]
    gazey = [y - ymin for y in gazey]

    return img, bbox, gazex, gazey, depth
    

def get_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    # Adapted from https://github.com/ejcgt/attention-target-detection/blob/master/utils/imutils.py

    img = torch.zeros(height, width)
    if gazex < 0 or gazey < 0:  # return empty map if out of frame
        return img
    gazex = int(gazex * width)
    gazey = int(gazey * height)

    # Check that any part of the gaussian is in-bounds
    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if htype == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif htype == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / img.max()  # normalize heatmap so it has max value of 1
    return img