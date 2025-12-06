import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np

import argparse
import logging
import os
from models import DiT_models
from transport import create_transport
from torch.autograd import grad
from download import find_model
from train import get_dataset


def make_x0_bank(M, x, seed=2025):
    g = torch.Generator(device=x.device).manual_seed(seed)
    bank = torch.randn((M, *x.shape[1:]), generator=g, device=x.device, dtype=x.dtype)
    m = M // 2
    bank[m:2*m] = -bank[:m]
    return bank  


def grad_score(model, diffusion, dataloader, scale_factor, device, num_t_steps=8, M=4, seed=0):
    ema_alpha=0.9
    x0_bank = None       

    model_params = [p for p in model.parameters() if p.requires_grad]
    t_shared = None
    ema_per_t = None
    gradients = {}
    for l, batch in enumerate(dataloader):

        x = batch['image']
        y = batch['label']
        sample_id = batch['id'].item()
        
        y[...] = 0

        x = x.to(device)*  scale_factor
        y = y.to(device)
        model_kwargs = dict(y=y)

        if t_shared is None:
            t_shared = torch.linspace(0, 1, steps=num_t_steps + 2, device=device, dtype=x.dtype)[1:-1]
            ema_per_t = torch.ones_like(t_shared, dtype=torch.float64)
            x0_bank = make_x0_bank(M, x)
        cumulative_grad_norm = 0
        for k, tk in enumerate(t_shared):
            t_vec = tk.expand(x.size(0)).to(dtype=x.dtype)
            g2_sum = 0.0
            for m in range(M):
                x0_m = x0_bank[m].unsqueeze(0)
                loss_dict = diffusion.training_losses(model, x, t=t_vec, x0=x0_m, model_kwargs=model_kwargs)
                loss = loss_dict['loss'].mean()

                grads = grad(loss, model_params, create_graph=False, allow_unused=True)
                g2 = sum((gi.float().detach()**2).sum() for gi in grads if gi is not None).item()
                g2_sum += g2
            g2_mean = g2_sum / M
            normed = g2_mean / (float(ema_per_t[k]) + 1e-8)
            cumulative_grad_norm += normed
            ema_per_t[k] = ema_alpha * ema_per_t[k] + (1.0 - ema_alpha) * g2_mean

        gradients[sample_id] = cumulative_grad_norm / len(t_shared)

    return gradients

def main(args):
    """
    score samples using an early epoch based on the gradient or loss signal.
    """

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_path = args.experiment_dir 
    checkpoint_dir =  f"{exp_path}/checkpoints" 
    score_dir = f"{exp_path}/scores"  # Stores saved model checkpoints
    print('score dir is %s' %score_dir)
    os.makedirs(score_dir, exist_ok=True)
    # Setup an experiment folder:
    dataset_name = args.dataset_name
    print('1 dataset name is %s' %dataset_name)

    diffusion = create_transport()
    learn_sigma = False

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
   
   
    in_channels = 4
    scale_factor = 0.8077
    num_classes = args.num_classes
    class_dropout_prob = 0
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob = class_dropout_prob, 
        learn_sigma = learn_sigma
    ).to(device)


    dataset = get_dataset(name=dataset_name)
    print('finished building dataset')
    end_index = min(args.start_index+30000, len(dataset))
    selected_index = np.arange(args.start_index, end_index)
    dataset = torch.utils.data.Subset(dataset, selected_index) 
    print('dataset length is %d' %len(dataset), 'start index is %d' %args.start_index)
    print('creating dataloader')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model_path = os.path.join(checkpoint_dir, '0030000.pt') 


    state_dict = find_model(model_path)
    model.load_state_dict(state_dict)
    model.eval() 

    grad_scores = grad_score(model, diffusion, dataloader, scale_factor, device, seed=args.seed)
    score_path = os.path.join(score_dir, 'score_grad_iter30k_%d_10t.pth'%args.start_index)
    torch.save(grad_scores, score_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="results/celebhq_unconditional_fm_unpruned_s2")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dataset_name", type=str, default='celebhq_precomputed')
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae_path", type=str, default='')   
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1_0000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()
    main(args)
