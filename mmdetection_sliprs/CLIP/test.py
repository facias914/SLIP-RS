import torch
import clip
from PIL import Image
import os


text_list_0 = [
    "a plane from a top-down view", 
    "a baseball-diamond from a top-down view",
    "a bridge from a top-down view",
    "a ground-track-field from a top-down view",
    "a small-vehicle from a top-down view",
    "a large-vehicle from a top-down view",
    "a ship from a top-down view",
    "a tennis-court from a top-down view",
    "a basketball-court from a top-down view",
    "a storage-tank from a top-down view",
    "a soccer-ball-field from a top-down view",
    "a roundabout from a top-down view",
    "a harbor from a top-down view",
    "a swimming-pool from a top-down view",
    "a helicopter from a top-down view",
    "a container-crane from a top-down view",
    "a airport from a top-down view",
    "a helipad from a top-down view",
    "background"
]

# text_list_1 = [
#     "a plane from a bird's-eye view", 
#     "a baseball-diamond from a bird's-eye view",
#     "a bridge from a bird's-eye view",
#     "a ground-track-field from a bird's-eye view",
#     "a small-vehicle from a bird's-eye view",
#     "a large-vehicle from a bird's-eye view",
#     "a ship from a bird's-eye view",
#     "a tennis-court from a bird's-eye view",
#     "a basketball-court from a bird's-eye view",
#     "a storage-tank from a bird's-eye view",
#     "a soccer-ball-field from a bird's-eye view",
#     "a roundabout from a bird's-eye view",
#     "a harbor from a bird's-eye view",
#     "a swimming-pool from a bird's-eye view",
#     "a helicopter from a bird's-eye view",
#     "a container-crane from a bird's-eye view",
#     "a airport from a bird's-eye view",
#     "a helipad from a bird's-eye view",
# ]
save_path = "/liyunheng/WCX/mmdetection-main/pretrained_weights/bird's-eye-view.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load("/liyunheng/WCX/mmdetection-main/pretrained_weights/ViT-B-32.pt", device=device)

image = preprocess(Image.open("/liyunheng/WCX/mmdetection-main/CLIP-main/tile_00573.png")).unsqueeze(0).to(device)
text = clip.tokenize(text_list_0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

torch.save(text_features, save_path)
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]