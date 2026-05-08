import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import itertools


ATTR_DICT = {"Plane" : {'Engine position': ['At wing roots and lower fuselage', 'Beneath the wings',
                                            'On the nose', 'Rear fuselage', 'Above the wings', 'Embedded within wing'],
                        'Number of engines': ['Eight-engine', 'Four-engine', 'One-engine', 'Twin-engine', 'Ten-engine'],
                        'Propulsion type': ['Jet', 'Propeller'],
                        'Purpose': ['AerialSupport Aircraft', 'Airborne Early Warning Aircraft', 'Airline Aircraft',
                                    'Anti-Submarine Warfare Aircraft', 'Bomber', 'Chartered aircraft', 'Fighter',
                                    'Propeller', 'Trainer', 'Transport Aircraft', 'Attack aircraft'],
                        'Usage': ['Civilian Aircraft', 'Commercial Aircraft', 'Military Aircraft'],
                        'Wing configuration': ['Straight wing', 'Swept delta wing', 'Swept diamond-like wing',
                                                'Swept wing', 'Swept, variable-sweep wing', 'Flying wing']},
            "Ship" : {'Usage': ['Civilian Ship', 'Commercial Ship', 'Engineering Ship', 'Military Ship'],
                        'Subcat': ['Barge', 'Container Ship', 'Dry Cargo Ship',
                                    'Cruise Ship', 'Liquid Cargo Ship', 'RoRo', 'Yacht'],
                        'Purpose': ['Aircraft Carrier', 'Amphibious Ship', 'Auxiliary Ship', 'Cargo Ship', 'Commander', 
                                    'Cruiser', 'Destroyer', 'Frigate', 'Landing', 'Medical Ship', 'Military Transport Ship', 
                                    'Passenger Ship', 'Patrol', 'Submarine', 'Test ship', 'Training ship', 'Tugboat',
                                    'Fishing Vessel', 'Motorboat']},
            "Vehicle" : {'Purpose': ['Bus', 'Cargo Truck', 'Dump Truck', 'Excavator', 'Pick-up', 'Small Passenger Car',
                                     'Tractor', 'Truck Tractor', 'Van'],
                         'Usage': ['Engineering Vehicle', 'Large Civilian Vehicle', 'Small Civilian Vehicle', 'Truck']}}


TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    # transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    # transforms.CenterCrop(224),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    # ], p=0.2),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


class AttributeContrastiveBuilder:
    def __init__(self, drop_prob=0.5, shuffle_prob=1.0, max_negatives=40, use_value=True):
        self.attr_dict = ATTR_DICT
        self.drop_prob = drop_prob
        self.shuffle_prob = shuffle_prob
        self.max_negatives = max_negatives
        self.use_value = use_value

    def build_positive_text(self, attr_sample):
        proto = attr_sample["ProtoTag"]
        attrs = [(k, v) for k, v in attr_sample.items() if k != "ProtoTag"]

        # random drop
        attrs = [(k, v) for k, v in attrs if random.random() > self.drop_prob]

        # random shuffle
        if random.random() < self.shuffle_prob:
            random.shuffle(attrs)

        if self.use_value:
            parts = [f"{v}" for _, v in attrs]
        else:
            parts = [f"{k}" for k, _ in attrs]

        text = f"{proto} + " + " + ".join(parts) if parts else proto
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

    def __call__(self, attr_sample):
        pos_text, attrs = self.build_positive_text(attr_sample)
        neg_text = self.build_negative_text(attr_sample["ProtoTag"], attrs)
        return pos_text, neg_text


class FGRSDataset(Dataset):
    def __init__(self, image_root, label_root, transform=TRANSFORM):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform

        self.text_builder = AttributeContrastiveBuilder(max_negatives=15)

        self.data_dict = self._build_data_dict()
        self.image_paths = list(self.data_dict.keys())

    def _build_data_dict(self):
        data_dict = {}

        for root, dirs, files in os.walk(self.label_root):
            for file in files:
                if not file.endswith(".json"):
                    continue

                json_path = os.path.join(root, file)
                with open(json_path, "r") as f:
                    label_data = json.load(f)

                relative_dir = os.path.relpath(root, self.label_root)
                if relative_dir == '.':
                    relative_dir = os.path.basename(json_path).split('.')[0]
                else:
                    relative_dir = relative_dir + '/' + os.path.basename(json_path).split('.')[0]

                for img_name, attr_dict in label_data.items():
                    img_path = os.path.join(self.image_root, relative_dir, img_name)
                    if os.path.exists(img_path):
                        data_dict[img_path] = attr_dict
                    else:
                        print(f"Warning: {img_path} does not exist!")

        return data_dict

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        attr_dict = self.data_dict[img_path]

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        pos_text, neg_text = self.text_builder(attr_dict)

        return img, pos_text, neg_text


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = FGRSDataset(
        image_root="/data/datasets/RemoteClip/train/image/Plane",
        label_root="/data/datasets/RemoteClip/train/label/Plane",
        transform=TRANSFORM
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4)

    batch = next(iter(dataloader))

    print()
