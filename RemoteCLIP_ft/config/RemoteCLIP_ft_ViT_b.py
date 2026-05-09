# model
clip_weight = "./pretrain_weights/ViT-B-32.pt"
pretrain_weight = "./pretrain_weights/RemoteCLIP-ViT-B-32.pt"

# lora
backbone = "ViT-B/32"
encoder = "both"
position = "all"
params = ['q', 'k', 'v']
r = 2
alpha = 1
dropout_rate = 0.2

# dataset
image_root_list=["./DATA/RS_Attri_Cls/train/image/Plane", 
                 "./DATA/RS_Attri_Cls/train/image/Ship",
                 "./DATA/RS_Attri_Cls/train/image/Vehicle"]
label_root_list=["./DATA/RS_Attri_Cls/train/label/Plane", 
                 "./DATA/RS_Attri_Cls/train/label/Ship",
                 "./DATA/RS_Attri_Cls/train/label/Vehicle"]


val_image_root="./DATA/RS_Attri_Cls/test/image/Plane"
val_label_root="./DATA/RS_Attri_Cls/test/label/Plane"
test_attris = ["Usage", "Purpose", "Engine position", "Number of engines", "Propulsion type", "Wing configuration"]

# learner
lr = 0.00001
batch_size = 32
epochs = 100
save_checkpoint_interval = 5