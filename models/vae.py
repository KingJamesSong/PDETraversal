import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder_layer_sizes = encoder_layer_sizes[0]
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)

    def forward(self, x):

        if x.dim() > 2:
            x = x.view(-1, self.encoder_layer_sizes)

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z):

        recon_x = self.decoder(z)

        return recon_x


class ConvVAE(nn.Module):

    def __init__(self, num_channel, latent_size, img_size):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = ConvEncoder2(n_cin=num_channel,s_dim=latent_size,n_hw=img_size)
        self.decoder = ConvDecoder2(n_cout=num_channel,s_dim=latent_size,n_hw=img_size)
        #self.encoder = ConvEncoder(num_channel=num_channel,latent_size=latent_size)
        #self.decoder = ConvDecoder(latent_size=latent_size)

    def forward(self, x):

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z):

        recon_x = self.decoder(z)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()

        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)

        return x

class ConvEncoder(nn.Module):

    def __init__(self, num_channel, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
        )

        self.linear_means = nn.Linear(256, latent_size)
        self.linear_log_var = nn.Linear(256, latent_size)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(-1, 256)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

class ConvDecoder(nn.Module):

    def __init__(self, latent_size):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # B, nc, 64, 64
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder(z)

        return x

class ConvEncoder2(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw):
        super().__init__()

        self.s_dim = s_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw//4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 2, kernel_size=1, stride=1, padding=0),
        )

        self.linear_means = nn.Linear(s_dim * 2, s_dim)
        self.linear_log_var = nn.Linear(s_dim * 2, s_dim)

    def forward(self, x):

        x = self.encoder(x)
        x = x.view(-1, self.s_dim * 2)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

class ConvDecoder2(nn.Module):

    def __init__(self, s_dim, n_cout, n_hw):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(s_dim, s_dim),  # B, 256
            View((-1, s_dim, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim * 2,kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 2, s_dim * 3,kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 3, s_dim * 2 ,kernel_size=(n_hw//4), stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 2, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, n_cout, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.decoder(z)

        return x

