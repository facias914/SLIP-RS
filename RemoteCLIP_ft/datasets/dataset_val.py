import os
import json
from PIL import Image
from torch.utils.data import Dataset


class AttrBasedClassificationDataset(Dataset):
    def __init__(self, image_root, label_root, test_attri, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.test_attri = test_attri
        self.transform = transform

        self.samples = [] 
        self.classes = [] 
        self.cls2idx = {} 

        for cls_folder in sorted(os.listdir(image_root)):
            img_dir = os.path.join(image_root, cls_folder)
            json_path = os.path.join(label_root, f"{cls_folder}.json")

            if not os.path.isdir(img_dir) or not os.path.exists(json_path):
                print(f"Warning: missing folder/json for {cls_folder}")
                continue

            with open(json_path, "r") as f:
                label_dict = json.load(f)

            for img_name, attr_dict in label_dict.items():
                img_path = os.path.join(img_dir, img_name)
                if not os.path.exists(img_path):
                    print(f"Warning: {img_path} not found!")
                    continue

                if self.test_attri not in attr_dict:
                    print(f"Warning: {self.test_attri} missing in {img_name}")
                    continue

                attr_value = attr_dict[self.test_attri]
                if attr_value not in self.cls2idx:
                    self.cls2idx[attr_value] = len(self.classes)
                    self.classes.append(attr_value)

                label_idx = self.cls2idx[attr_value]
                self.samples.append((img_path, label_idx))

        print(f"Total images: {len(self.samples)}")
        print(f"Classes ({len(self.classes)}): {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

