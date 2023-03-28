"""
Code for generating the paired images for calculate VP metric.

For example, the following command works:

python gen_pairs.py
--model_path path_to_OroJaR_netG_model
--model_name OroJaR
--model_type gan
"""

import argparse
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
import os.path as osp
import json
from torch.nn import functional as F


class ModelArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def build_gan(gan_type, target_classes, stylegan2_resolution, shift_in_w_space, use_cuda, multi_gpu):
    # -- BigGAN
    if gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                         target_classes=target_classes)
    # -- ProgGAN
    elif gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]])
    # -- StyleGAN2
    elif gan_type == 'StyleGAN2':
        G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][stylegan2_resolution],
                            resolution=stylegan2_resolution,
                            shift_in_w_space=shift_in_w_space)
    # -- Spectrally Normalised GAN (SNGAN)
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[gan_type]['weights'][GAN_RESOLUTIONS[gan_type]],
                        gan_type=gan_type)
    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Parallelize GAN generator model into multiple GPUs if possible
    if multi_gpu:
        G = DataParallelPassthrough(G)

    return G

def sample_z(batch_size, dim_z, truncation=None):
    """Sample a random latent code from multi-variate standard Gaussian distribution with/without truncation.

    Args:
        batch_size (int)   : batch size (number of latent codes)
        dim_z (int)        : latent space dimensionality
        truncation (float) : truncation parameter

    Returns:
        z (torch.Tensor)   : batch of latent codes
    """
    if truncation is None or truncation == 1.0:
        return torch.randn(batch_size, dim_z)
    else:
        return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=(batch_size, dim_z))).to(torch.float)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the Disentanglement of ProgressiveGAN')

    parser.add_argument('-v', '--verbose', action='store_true', help="set verbose mode on")
    # ================================================================================================================ #
    parser.add_argument('--exp', type=str, required=True, help="set experiment's model dir (created by `train.py`)")
    parser.add_argument('--shift-steps', type=int, default=16, help="set number of shifts per positive/negative path "
                                                                    "direction")
    parser.add_argument('--eps', type=float, default=0.2, help="set shift step magnitude")
    parser.add_argument('--shift-leap', type=int, default=1,
                        help="set path shift leap (after how many steps to generate images)")
    parser.add_argument('--batch-size', type=int, help="set generator batch size (if not set, use the total number of "
                                                       "images per path)")
    parser.add_argument('--img-size', type=int, help="set size of saved generated images (if not set, use the output "
                                                     "size of the respective GAN generator)")
    parser.add_argument('--img-quality', type=int, default=75, help="set JPEG image quality")
    parser.add_argument('--gif', action='store_true', help="Create GIF traversals")
    parser.add_argument('--gif-size', type=int, default=256, help="set gif resolution")
    parser.add_argument('--gif-fps', type=int, default=30, help="set gif frame rate")
    # ================================================================================================================ #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    parser.add_argument('--sefa', default=False, type=str2bool,
                        help='Use SeFa on the first conv/fc layer to achieve disentanglement.')
    parser.add_argument('--save_dir', type=str, default='./pairs', help='figures are saved here')
    #parser.add_argument('--sample_dir', type=str, default='./samples', help='figures are saved here')

    opt = parser.parse_args()

    # Parse given arguments
    args = parser.parse_args()

    # Check structure of `args.exp`
    if not osp.isdir(args.exp):
        raise NotADirectoryError("Invalid given directory: {}".format(args.exp))

    # -- args.json file (pre-trained model arguments)
    args_json_file = osp.join(args.exp, 'args.json')
    if not osp.isfile(args_json_file):
        raise FileNotFoundError("File not found: {}".format(args_json_file))
    args_json = ModelArgs(**json.load(open(args_json_file)))
    gan_type = args_json.__dict__["gan_type"]

    # -- models directory (support sets and reconstructor, final or checkpoint files)
    models_dir = osp.join(args.exp, 'models')
    if not osp.isdir(models_dir):
        raise NotADirectoryError("Invalid models directory: {}".format(models_dir))

    # ---- Get all files of models directory
    models_dir_files = [f for f in os.listdir(models_dir) if osp.isfile(osp.join(models_dir, f))]

    # ---- Check for support sets file (final or checkpoint)
    support_sets_model = osp.join(models_dir, 'checkpoint.pt')
    if not osp.isfile(support_sets_model):
        support_sets_checkpoint_files = []
        for f in models_dir_files:
            if 'support_sets-' in f:
                support_sets_checkpoint_files.append(f)
        support_sets_checkpoint_files.sort()
        print(models_dir, support_sets_checkpoint_files)
        support_sets_model = osp.join(models_dir, support_sets_checkpoint_files[-1])

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

    netG = build_gan(gan_type=gan_type,
                  target_classes=args_json.__dict__["biggan_target_classes"],
                  stylegan2_resolution=args_json.__dict__["stylegan2_resolution"],
                  shift_in_w_space=args_json.__dict__["shift_in_w_space"],
                  use_cuda=use_cuda,
                  multi_gpu=multi_gpu).eval()

    S = WavePDE(num_support_sets=args_json.__dict__["num_support_sets"],
                num_support_dipoles=args_json.__dict__["num_support_dipoles"],
                support_vectors_dim=netG.dim_z,
                learn_alphas=args_json.__dict__["learn_alphas"],
                learn_gammas=args_json.__dict__["learn_gammas"],
                gamma=1.0 / netG.dim_z if args_json.__dict__["gamma"] is None else args_json.__dict__["gamma"])
    # For stylegan remove the last activation layer otherwise the changes are too small
    #print(gan_type)
    if gan_type == 'StyleGAN2':
        print("StyleGAN2 Loaded")
        for i in range(S.num_support_sets):
            S.MLP_SET[i].activation4 = nn.Identity()

    if opt.sefa:
        print(netG.G)
        weight = netG.G[0].weight.data
        u, s, v = torch.svd(weight)
        print(weight.size())
        inter_directions = u[:S.num_support_sets,:]
    #    weight_name = 'layers.0.layers.conv.weight'
    #    weight = state_dict[weight_name]
    #    size = weight.size()
    #    weight = weight.reshape((weight.size(0), -1)).T
    #    U, S, V = torch.svd(weight)
    #    new_weight = U * S.unsqueeze(0).repeat((U.size(0), 1))
    #    state_dict[weight_name] = new_weight.T.reshape(size)

    S.eval()

    # Upload support sets model to GPU
    if use_cuda:
        S = S.cuda()

    # Set number of generative paths
    num_gen_paths = S.num_support_sets

    # Create output dir for generated images
    #out_dir = osp.join(args.exp, 'vp_pairs', args.pool,
    #                   '{}_{}_{}'.format(2 * args.shift_steps, args.eps, round(2 * args.shift_steps * args.eps, 3)))
    #os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(args.exp, 'vp_pairs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    nz = num_gen_paths
    n_samples = 40000
    batch_size = 2
    n_batches = n_samples // batch_size
    print(S.num_support_dipoles)

    for i in range(n_batches):
        print('Generating image pairs %d/%d ...' % (i, n_batches))
        grid_labels = np.zeros([batch_size, 0], dtype=np.float32)

        z_1 = torch.randn(batch_size, netG.dim_z).cuda()

        #idx = np.array(list(range(100)))  # full

        #delta_dim = np.random.randint(0, nz, size=[batch_size])
        #delta_dim = idx[delta_dim]
        delta_dim = torch.randint(0,S.num_support_sets,(1,1),requires_grad=False)


        if opt.sefa:
            delta_z = inter_directions[delta_dim.squeeze(),:].repeat(batch_size,1)
            z_shifted = z_1 + delta_z
        else:
            if args_json.__dict__["shift_in_w_space"]:
                z_1 = netG.get_w(z_1)
            z_shifted = z_1.clone()
            for step in range(S.num_support_dipoles):
                _, shift = S.inference(delta_dim, z_shifted, step * torch.ones(1, 1, requires_grad=True), netG)
                z_shifted = z_shifted + shift
            #energy,z_shifted, z_shifted2 ,_ = S(delta_dim, z_1, (S.num_support_dipoles/2-1) * torch.ones(1, 1, requires_grad=True), netG)

        delta_onehot = np.zeros((batch_size, nz))
        delta_onehot[:, delta_dim.squeeze()] = 1

        if i == 0:
            labels = delta_onehot
        else:
            labels = np.concatenate([labels, delta_onehot], axis=0)
        fakes_1 = netG(z_1)
        fakes_2 = netG(z_shifted)
        fakes_1 = F.interpolate(
            fakes_1, size=(256, 256), mode="bilinear", align_corners=False
        )
        fakes_2 = F.interpolate(
            fakes_2, size=(256, 256), mode="bilinear", align_corners=False
        )
        for j in range(fakes_1.shape[0]):
            img_1 = fakes_1[j, torch.LongTensor([2, 1, 0]), :, :]
            img_2 = fakes_2[j, torch.LongTensor([2, 1, 0]), :, :]
            img_1 = img_1.cpu().detach().numpy().transpose((1, 2, 0))
            img_2 = img_2.cpu().detach().numpy().transpose((1, 2, 0))
            pair_np = np.concatenate([img_1, img_2], axis=1)
            img = (pair_np + 1) * 127.5
            #sample = (img_1 + 1) * 127.5
            cv2.imwrite(
                os.path.join(out_path,
                             'pair_%06d.jpg' % (i * batch_size + j)), img)
            #cv2.imwrite(
            #    os.path.join(sample_path,
            #                 'sample_%06d.jpg' % (i * batch_size + j)), sample)

    np.save(os.path.join(out_path, 'labels.npy'), labels)