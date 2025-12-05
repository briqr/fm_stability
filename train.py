# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT-FM using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
from torch.utils.data import Subset
from image_folder import ImageFolderFilenames
from models import DiT_models
from retrain import random_opt, score_based_ind
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from train_autoencoder import get_autoencoder
from torchvision.utils import save_image
import datasets
from datasets.transforms import build_transforms
from datasets import build_dataset, force_flip_then
from retrain_cluster import furthestmid_cluster, equal_cluster, mid_cluster, random_cluster, drop_clusters
from transport import create_transport
import logging
from train_consts import *
import torch
from automotive.nuimagedataset import NuImagesCustomDataset, NUSCENES_DATAROOT
from automotive.cityscapedataset import CityscapesCoarseDataset, CITYSCAPES_DATAROOT
from ffhq.ffhqdataset import FFHQDataset, FFHQ_DATAROOT 
torch.autograd.set_detect_anomaly(True)
import json
import re
#################################################################################
#                             Training Helper Functions                         #
#################################################################################



#################################################################################
#                                  Training Loop                                #
#################################################################################



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#splits: "train", "validation", "test"

#splits: "train", "validation", "test"
def get_dataset(split='train', is_training = True, name ='celebhq'):
    if name == 'celebhq':
        cache_dir = 'DATASETS/'
        dataset_name = 'jxie/celeba-hq'
    
    train_data_config = {'path': dataset_name,
                    'cache_dir': cache_dir, 
                    'split': split,
                    'trust_remote_code': True                    }

    transforms_config = {'no_aug': True,
                            'is_training': is_training,
                            'input_size': 256,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]
                            }

    data_transform = build_transforms(transforms_config, dataset_name=dataset_name)
    print('building dataset')
    dataset = build_dataset(
        train_data_config,
        transforms=data_transform,
    )
    return dataset


def to_cuda_long(x, device):
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(x)
    t = t.to(device=device, dtype=torch.long, non_blocking=True)
    
    return t

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    dataset_name = args.dataset_name
    
    pr = args.pruning_ratio
    is_pruned = pr > 0.00001
    method = args.pruning_method
    inverse = args.inverse > 0.00001
    transport = create_transport()
    learn_sigma = False

    in_channels = 4 
    scale_factor = 0.8077
    num_classes = args.num_classes
    
    num_clusters = 24
    cluster_path = args.clusters_path%num_clusters
    vae = get_autoencoder(args.vae_path).to(device).eval() #
    
   
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        pr_str = str(pr).replace('.', '')
        method_str = method
        if inverse > 0:
            method_str =  'inverse' + method
        
        if args.resume:
            exp = args.resume.split('/')[1]
            experiment_dir = f"{args.results_dir}/{exp}"
            match = re.search(r'_pruned(\d+)', exp)
            pr = float(match.group(1))/100
            print('*****pr from resume path', pr) 


        experiment_dir = f"{args.results_dir}/%s_%s_pruned%s"%(dataset_name, method_str, pr_str)  # Create an experiment folder
        if not args.resume:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            experiment_dir = f"{experiment_dir}_seed{seed}_{experiment_index:03d}"
            experiment_dir = f"{experiment_dir}"
        selected_index_path = os.path.join(experiment_dir, 'selected_index.pth')
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        print(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8


    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob = 0, #do not use classifier-free guidance
        learn_sigma = learn_sigma
    )

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        logger.info(f"Using checkpoint: {args.resume}")
        print(f"Using checkpoint: {args.resume}")

    
    model = DDP(model.to(device), device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    if args.resume:
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print('args', args)
    logger.info(f"args: {args}")

    
    
    dataset = get_dataset(name=dataset_name)
    if is_pruned:
        if rank ==0:
            if method.startswith('loss'):
                scores = torch.load(args.score_dir, map_location='cpu')
                selected_index = score_based_ind(scores, 1-pr, largest=not inverse, is_mid=is_mid)
            elif method.startswith('grad'):
                scores = torch.load(args.score_dir, map_location='cpu')
                selected_index = score_based_ind(scores, 1-pr, largest=not inverse, is_mid=is_mid)
            elif method.startswith('random'):
                selected_index = random_opt(len(dataset), 1-pr)[1]
            
            elif method.startswith('cluster_furthest'): # pruning based on clustering   
                selected_index = furthestmid_cluster(1-pr, cluster_path, largest= True)
            elif method.startswith('cluster_nearest'): # pruning based on clustering   
                selected_index = furthestmid_cluster(1-pr, cluster_path, largest= False)
            elif method.startswith('cluster_random'): # pruning based on clustering   
                selected_index = random_cluster(1-pr,cluster_path)
            
            elif method.startswith('balanced_cluster_furthest'): # pruning based on clustering   
                selected_index = equal_cluster(cluster_path, 1-pr, largest= True)
            elif method.startswith('balanced_cluster_nearest'): # pruning based on clustering   
                selected_index = equal_cluster(cluster_path, 1-pr, largest= False)
            elif method.startswith('balanced_cluster_random'): # pruning based on clustering   
                selected_index = equal_cluster(cluster_path, 1-pr, is_random= True)
            
            elif method.startswith('drop_cluster'): 
                selected_index = drop_clusters(pr, cluster_path=cluster_path)
            elif method == 'gender_female':
                with open('gender_indices/train_female_indices.json', 'r') as f:
                    selected_index = json.load(f)
            elif method == 'gender_male':
                with open('gender_indices/train_male_indices.json', 'r') as f:
                    selected_index = json.load(f)
            
            #when we have randomness in sampling while using ddp, we want to make sure all gpus see the same subset
            selected_index = to_cuda_long(selected_index, device)
            k0 = torch.tensor(selected_index.numel(), device=device, dtype=torch.int64)
            
        else:
            k0 = torch.empty(1, device=device, dtype=torch.int64)
        dist.broadcast(k0, src=0)
        k = int(k0.item())
        if rank != 0:
            selected_index = torch.empty(k, device=device, dtype=torch.long)
        dist.broadcast(selected_index, src=0)
        if rank == 0:
                torch.save(selected_index.detach().cpu(), selected_index_path)
        selected_index = selected_index.cpu().tolist()
        if pr > 0.005:
            dataset = torch.utils.data.Subset(dataset, selected_index) 
    print('finished building dataset')

    print('creating dataloader')
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")
    print(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")

    train_steps = 0
    log_steps = 0
    running_loss = 0

    if args.resume:
        train_steps = int(args.resume.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / (len(dataset) / args.global_batch_size))
        logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
        print(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights 
        start_epoch = 0
    
    model.train()  
    ema.eval()  #


    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        print(f"Beginning epoch {epoch}...")
        global_pos = 0
        for batch in loader:
            x = batch['image']
            y = batch['label']
            with torch.no_grad():
                x = vae.encode(x)['quantized']
            y[...] = 0
            x = x.to(device)
            y = y.to(device)
            model_kwargs = dict(y=y)
            bs = x.shape[0]

            x = x.contiguous() * scale_factor
            
            
            loss_dict = transport.training_losses(model, x, model_kwargs)
            
            
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_pos += bs

            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    print(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--score_dir", type=str, default="")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_clusters", type=int, default=24)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1_0000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pruning_method", type=str, default='random')
    parser.add_argument("--clusters_path", type=str, default='')
    parser.add_argument("--pruning_ratio", type=float, default=0)
    parser.add_argument("--inverse", type=int, default=0)
    parser.add_argument("--transport", type=str, default='fm')
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--is_random", type=int, default=0)
    args = parser.parse_args()
    main(args)



