import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(144, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(144, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(144),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])


class ClsDataset(Dataset):
    def __init__(self, root_dir, label_dir=None, mode='train'):

        self.root_dir = root_dir
        self.mode = mode
        self.transform = TRANSFORM_TRAIN if mode == 'train' else TRANSFORM_TEST

        self.samples = []
        # self.classes = []
        self.classes = ['bus', 'car', 'van']
        self.class_to_idx = {}

        if mode == 'train':
            # self.classes = sorted([
            #     d for d in os.listdir(root_dir)
            #     if os.path.isdir(os.path.join(root_dir, d))
            # ])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(cls_dir, fname)
                        label = self.class_to_idx[cls_name]
                        self.samples.append((path, label))

        elif mode == 'test':
            txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            # self.classes = sorted([os.path.splitext(f)[0] for f in txt_files])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                txt_path = os.path.join(label_dir, f"{cls_name}.txt")
                with open(txt_path, 'r') as f:
                    for line in f:
                        rel_path = line.strip()
                        if not rel_path:
                            continue
                        abs_path = os.path.join(root_dir, rel_path)
                        self.samples.append((abs_path, self.class_to_idx[cls_name]))

        else:
            raise ValueError("mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label
