import os
import json
import time
import datetime
import argparse
import random
import os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, ConcatDataset
import torch.nn.functional as F
from util.slconfig import SLConfig
import util.misc as utils

from util.logger import setup_logger
from datasets.dataset import FGRSDataset, collate_fn
from datasets.dataset_val import AttrBasedClassificationDataset
import model.clip as clip
from model.loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters


def get_args_parser():
    parser = argparse.ArgumentParser('Set gaze detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', default=True, type=bool)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def get_model(model):
    return model.module if hasattr(model, "module") else model


def build_val_dataloader(args, preprocess, collate_fn):
    datasets_val = []
    data_loaders_val = []

    def make_loader(image_root, label_root, test_attri):
        dataset_val = AttrBasedClassificationDataset(image_root, label_root, test_attri, preprocess)
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
        data_loader_val = DataLoader(
            dataset_val,
            batch_size=64,
            sampler=sampler_val,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        return dataset_val, data_loader_val

    if isinstance(args.val_image_root, (list, tuple)):
        for txt_path in args.val_image_root:
            dataset_val, data_loader_val = make_loader(txt_path)
            datasets_val.append(dataset_val)
            data_loaders_val.append(data_loader_val)
    else:
        for test_attri in args.test_attris:
            dataset_val, data_loader_val = make_loader(args.val_image_root, args.val_label_root, test_attri)
            datasets_val.append(dataset_val)
            data_loaders_val.append(data_loader_val)

    return datasets_val, data_loaders_val


def evaluate_all(model_without_ddp, datasets_val, data_loaders_val, epoch, device, args, logger=None):
    if not isinstance(datasets_val, (list, tuple)):
        datasets_val = [datasets_val]
    if not isinstance(data_loaders_val, (list, tuple)):
        data_loaders_val = [data_loaders_val]

    for i, (dataset_val, data_loader_val) in enumerate(zip(datasets_val, data_loaders_val)):
        if len(datasets_val) > 1:
            print(f"\n=== Evaluating validation set {i+1}/{len(datasets_val)} ({len(dataset_val)} samples) ===")

        evaluate_(model_without_ddp, data_loader_val, dataset_val.classes, epoch, device, args=args,
                  logger=(logger if args.save_log else None))


def train_one_epoch(model, data_loader, optimizer, device, epoch, args, 
                    lr_scheduler=None, logger=None):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    epoch_total_loss = 0
    for cur_iter, batch in enumerate(data_loader):
        imgs, pos_text, neg_text = batch
        imgs = imgs.to(device)
        clip_model = get_model(model)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = clip_model.encode_image(imgs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            B, D = image_features.shape

            text_tokens = clip.tokenize(pos_text).to(device)
            pos_text_features = clip_model.encode_text(text_tokens)
            pos_text_features = pos_text_features / pos_text_features.norm(dim=1, keepdim=True)

            neg_text_features = []
            for negs in neg_text:
                if len(negs) == 0:
                    neg_text_features.append([])
                    continue
                neg_tokens = clip.tokenize(negs).to(device)
                feats = clip_model.encode_text(neg_tokens)  # [K, D]
                feats = feats / feats.norm(dim=1, keepdim=True)
                neg_text_features.append(feats)

        temperature = clip_model.logit_scale.exp()
        losses = []
        for i in range(B):
            vi = image_features[i]          # [D]
            ti_pos = pos_text_features[i]   # [D]
            sim_pos = temperature * vi @ ti_pos.t()

            if len(neg_text_features[i]) > 0:
                ti_negs = neg_text_features[i]             # [K, D]
                sim_negs = temperature * vi @ ti_negs.t()  # [K]
                logits = torch.cat([sim_pos.unsqueeze(0), sim_negs])  # [1 + K]
            else:
                continue

            labels = torch.zeros(1, dtype=torch.long, device=device)
            loss_i = F.cross_entropy(logits.unsqueeze(0), labels)
            losses.append(loss_i)

        loss = torch.stack(losses).mean()
        epoch_total_loss = epoch_total_loss + loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        msg = (f"TRAIN EPOCH {epoch}, "
                f"iter {cur_iter}/{len(data_loader)}, "
                f"loss={loss.item():.4f}, "
                f"lr={current_lr:.6f}")
        logger.info(msg)
        print(msg)

    loss_epoch = epoch_total_loss / len(data_loader)
    msg = (f"****TRAIN Epoch**** {epoch}, "
            f"loss={loss_epoch:.4f}")
    logger.info(msg)
    print(msg)


@torch.no_grad()
def evaluate_(model, data_loader_val, template, epoch, device, args, logger):
    model.eval()
    texts = template
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        texts = clip.tokenize(texts).cuda()
        class_embeddings = model.encode_text(texts)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    all_pred = []
    all_label = []
    for cur_iter, batch in enumerate(data_loader_val):
        imgs, label = batch
        imgs = imgs.to(device)
        label = torch.tensor(label, device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            image_features = model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        cosine_similarity = image_features @ text_features.t()
        all_pred.append(cosine_similarity)
        all_label.append(label)

    pred_list = utils.all_gather(all_pred)
    label_list = utils.all_gather(all_label)
    all_pred = torch.cat(pred_list[0], dim=0)
    all_label = torch.cat(label_list[0], dim=0)

    pred_label = all_pred.max(-1)[-1]
    all_auc = (pred_label == all_label).sum() / len(all_label)

    if utils.get_rank() == 0:
        msg = (f"****TEST EPOCH**** {epoch}, "
                f"Precision={all_auc:.6f}")
        logger.info(msg)
        print(msg)


def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
        
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CLIP
    model, preprocess = clip.load(args.clip_weight)
    if args.pretrain_weight is not None:
        ckpt = torch.load(args.pretrain_weight, map_location="cpu")
        message = model.load_state_dict(ckpt, strict=False)
        print(message)
    # model.eval()
    model.float()
    list_lora_layers = apply_lora(args, model)
    model = model.cuda() 
    # mark_only_lora_as_trainable(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # build dataset and dataloader
    datasets = []
    for img_root, lbl_root in zip(args.image_root_list, args.label_root_list):
        dataset = FGRSDataset(img_root, lbl_root)
        datasets.append(dataset)
        
    dataset_train = ConcatDataset(datasets)
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataset_val_list, data_loader_val_list = build_val_dataloader(args, preprocess, collate_fn)

    # build optimizer
    total_iters = len(data_loader_train) * args.epochs
    # optimizer = torch.optim.AdamW(get_lora_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])        

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(model, data_loader_train, optimizer, device, epoch, args, lr_scheduler, 
                        logger=(logger if args.save_log else None))

        evaluate_all(model_without_ddp, dataset_val_list, data_loader_val_list, epoch, device, args, 
                     logger=(logger if args.save_log else None))
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(weights, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
