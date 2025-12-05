
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from download import find_model
from models import DiT_models
import argparse
from autoencoder import get_autoencoder
import os
import json
from transport import create_transport
from transport import Sampler as transport_sampler
from tqdm import tqdm
import torch_fidelity

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_idx = json.load(open("data/imagenet_class_index.json"))
    print('model is ', args.ckpt)
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    dataset_name = args.dataset_name
    input2 = "datasets/%s/validation"%dataset_name
    
    
    in_channels = 4
    scale_factor = 0.8077
   
    class_dropout_prob = 0
    num_classes = args.num_classes
 
    results_dir = 'results' 
    ckpt_path = os.path.join(*[results_dir, args.ckpt, 'checkpoints', epoch])  

    state_dict = find_model(ckpt_path)
    exp_dir = ckpt_path.split('/')[1]
    epoch = ckpt_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(*[results_dir, exp_dir, 'vis', 'vis4000_%s_seed%d'%(epoch, args.seed)])

    
    samples_save_path = os.path.join(save_dir, 'gen_samples.pth')
    transport = create_transport()
    transport_sampler = transport_sampler(transport).sample_ode(
    sampling_method="euler",
    num_steps=60,
    )
    model_type = args.model

    print('***model type %s' % model_type)
    model = DiT_models[model_type](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob = class_dropout_prob,
        learn_sigma = False
    ).to(device)

    os.makedirs(save_dir, exist_ok=True)
    print('created dir %s'%save_dir)

    model.load_state_dict(state_dict)
    model.eval()  
    
 
    encoder_type = 'vq_gan_taming'
    encoder_path = args.vae_path
    vae = get_autoencoder(encoder_path, encoder_type).to(device) #

    vae.eval()
    batch_size = args.batch_size

    all_samples = []
    class_labels = torch.zeros(batch_size).int()
    y = class_labels.to(device)
    n = len(class_labels)
    for s in tqdm(range(args.num_iters)):
        z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
        model_kwargs = dict(y=y)
        forward_fn = model.forward
        samples = transport_sampler(
            z, forward_fn, **model_kwargs
        )[-1] 
        samples = vae.decode(samples / scale_factor)  
        samples = (((samples*0.5)+0.5)  *255).type(torch.uint8)
        all_samples.append(samples)

        all_samples = torch.cat(all_samples)

        samples_save_path = os.path.join(save_dir, 'all_samples_numsamples%d.pth')%len(all_samples)
        
        torch.save(all_samples, samples_save_path)

    metrics_dict = torch_fidelity.calculate_metrics(
    input1=all_samples, 
    input2=input2, 
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=False, 
    prc=True, 
    verbose=False
    )
    print(metrics_dict)
    print('model used', ckpt_path)
    print('number of samples %d' %len(all_samples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--epoch", type=str, default='0200000.pt') 
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--vae_path", type=str, default="")
    args = parser.parse_args()
    main(args)


