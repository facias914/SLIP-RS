TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 train.py \
    --output_dir logs/RemoteCLIP_ft_ViT_b -c config/RemoteCLIP_ft_ViT_b.py \
	--save_log True \