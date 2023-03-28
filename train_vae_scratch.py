import argparse
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
from models.vae import VAE, Encoder, ConvVAE, ConvEncoder, ConvEncoder2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from transforms import *
from dsprites import *
from Shapes3d import *
import torchvision

class DSprites(torch.utils.data.Dataset):

    def __init__(self,root, transform):
        super().__init__()
        data_dir = root
        self.data = np.load(os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')
        self.images = self.data['imgs']
        self.latents_values = self.data['latents_values']
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index:index+1]
        # to tensor
        img = torch.from_numpy(img.astype('float32'))
        # normalize
        #img = img.mul(2).sub(1)
        #img = self.transform(img)
        return img, self.latents_values[index]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

def main():

    parser = argparse.ArgumentParser(description="PotentialFlow training script")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-support-timesteps', type=int, help="set number of timesteps per potential")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-type', type=str, default='ResNet',
                        help='set reconstructor network type')
    parser.add_argument('--reconstructor-lr', type=float, default=1e-4,
                        help="set learning rate for reconstructor R optimization")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=1.00, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.00, help="regression loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument("--dsprites", type=bool, default=False)
    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    # === DSPRITES or MNIST ===
    if args.dsprites == True:
        G = ConvVAE(num_channel=1,latent_size=15*15+1,img_size=64)
        print("Intialize DSPRITES VAE")
    else:
        G = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28)
        print("Intialize MNIST VAE")

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Support Timesteps : {}".format(args.num_support_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.latent_size))

    S = WavePDE(num_support_sets=args.num_support_sets,
                    num_support_timesteps=args.num_support_timesteps,
                    support_vectors_dim=G.latent_size)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")


    if args.dsprites == True:
        R = ConvEncoder2(n_cin=2, s_dim=15*15+1,n_hw=64)
    else:
        R = ConvEncoder2(n_cin=6, s_dim=18 * 18, n_hw=28)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    config = {
        'wandb_on': False,
        'lr': 1e-4,
        'momentum': 0.9,
        'max_epochs': 100,
        'eval_epochs': 5,
        'dataset': 'DSprites',
        'seq_transforms': ['posX', 'posY', 'orientation', 'scale','shape'],
        'avail_transforms': ['posX', 'posY', 'orientation', 'scale', 'shape'],
        'seed': 1,
        'n_caps': 15,
        'cap_dim': 15,
        'n_transforms': args.num_support_timesteps // 2,
        'max_transform_len': 30,
        'mu_init': 30.0,
        'n_off_diag': 0,
        'group_kernel': (10, 10, 1),
        'n_is_samples': 10
    }
    if args.dsprites:
        print("DSPRITES DATASET LOADING")
        data_loader = get_dataloader(dir='/data/ysong',
                                     seq_transforms=config['seq_transforms'],
                                     avail_transforms=config['avail_transforms'],
                                     seq_len=config['n_transforms'],
                                     max_transform_len=config['max_transform_len'],
                                     batch_size=args.batch_size)
        trn = TrainerVAEScratchDsprites(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader)
    else:
        print("MNIST DATASET LOADING")
        dataset = MNIST(root='/nfs/data_lambda/ysong/', train=True, transform=transforms.ToTensor(),download=False)
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerVAEScratch(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader)

    # Training and Evaluation
    trn.train(generator=G, support_sets=S, reconstructor=R)
    trn.eval(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()
