import os
import time
import torch
import argparse
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from vae import VAE,ConvVAE
import torch.nn.functional as F
from transforms import *
angle_set = [0, 20, 40, 60, 80, 100, 120 , 140 ,160]
color_set = [180, 200, 220, 240, 260, 280 ,300 , 320 ,340]
scale_set = [1.0, 1.1, 1.2, 1,3, 1.4, 1.5 , 1.6 , 1.7, 1.8]
mnist_trans = AddRandomTransformationDims(angle_set=angle_set,color_set=color_set,scale_set=scale_set)
mnist_color = To_Color()

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


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    if args.dsprites:
        print("DSPRITES DATASET LOADING")
        dataset = DSprites(root='/nfs/data_chaos/ysong/simplegan_experiments/dataset',transform=transforms.ToTensor())
        #vae = ConvVAE(num_channel=1,latent_size=256).to(device)
        vae = ConvVAE(num_channel=1, latent_size=15 * 15 + 1, img_size=64).to(device)
    else:
        print("MNIST DATASET LOADING")
        dataset = MNIST(root='/data/ysong', train=True, transform=transforms.ToTensor(),download=True)
        vae = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28).to(device)
        #vae = VAE(
        #    encoder_layer_sizes=args.encoder_layer_sizes,
        #    latent_size=args.latent_size,
        #    decoder_layer_sizes=args.decoder_layer_sizes
        #).to(device)

    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)#, generator=torch.Generator(device='cuda'))

    def loss_fn(recon_x, x, mean, log_var):
        if args.dsprites==True:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(args.batch_size, -1), x.view(args.batch_size, -1), reduction='sum')
        else:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, args.encoder_layer_sizes[0]), x.view(-1, args.encoder_layer_sizes[0]), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)


    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            #x, y = x.to(device), y.to(device)
            if args.dsprites:
                x = x.to(device)
            else:
                x = mnist_color(x).to(device)
            #print(x.size())

            recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                #tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
                if args.dsprites:
                    torch.save(vae.state_dict(), 'vae_dsprites_conv_new.pt')
                else:
                    torch.save(vae.state_dict(), 'vae_mnist_conv3.pt')
                #z = torch.randn([10, args.latent_size]).to(device)
                #x = vae.inference(z)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list,default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list,default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--dsprites", type=bool, default=False)


    args = parser.parse_args()

    main(args)