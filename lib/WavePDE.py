import torch
from torch import nn
import math
from torch.autograd import grad
from torch.autograd.functional import jvp


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings,self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = (self.dim) // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP,self).__init__()

        self.layer_x = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        #Time position embedding used in Transformers and Diffusion Models
        self.layer_pos = SinusoidalPositionEmbeddings(n_in)
        self.layer_time = nn.Linear(n_in, n_in)
        self.activation2 = nn.GELU()
        self.layer_time2 = nn.Linear(n_in, n_in)

        self.layer_fusion = nn.Linear(n_in, n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = nn.Linear(n_in, n_out)
        self.activation4 = nn.Tanh()

    def forward(self, x, time):

        x = self.layer_x(x)
        x = self.activation1(x)

        time = self.layer_pos(time)
        time = self.layer_time(time)
        time = self.activation2(time)
        time = self.layer_time2(time)

        x = self.layer_fusion(x+time)
        x = self.activation3(x)
        x = self.layer_out(x)
        x = self.activation4(x)
        return x


class WavePDE(nn.Module):
    def __init__(self, num_support_sets, num_support_timesteps, support_vectors_dim):
        """

        Args:
            num_support_sets (int)    : number of MLPs
            num_support_timesteps (int) : number of timestamps
            support_vectors_dim (int) : dimensionality of latent code
            learn_alphas (bool)       :
            learn_gammas (bool)       :
            gamma (float)             :
        """
        super(WavePDE, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_timesteps = num_support_timesteps
        self.support_vectors_dim = support_vectors_dim
        self.c = nn.Parameter(torch.ones(num_support_sets,1,requires_grad=True))

        self.MLP_SET= nn.ModuleList([MLP(n_in=support_vectors_dim,n_out=1) for i in range(num_support_sets)])

    #Loss of initial condition
    def loss_ic(self,mlp,z):
        z = z.clone().requires_grad_()
        #z.requires_grad = True
        u = mlp(z,torch.FloatTensor([0]).to(z))
        u_z = grad(u.sum(), z, create_graph=True)[0]
        mse_0 =  torch.mean(torch.square(u_z))
        return mse_0

    #Loss of PDE
    def loss_pde(self,mlp,z,t,c,generator):
        z = z.clone().requires_grad_()
        #z.requires_grad = True
        t = t.clone().requires_grad_()
        #t.requires_grad = True

        u = mlp(z, t)
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_tt = grad(u_t.sum(), t, create_graph=True)[0]

        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]

        #PDE
        pde = u_tt - (c**2) * (u_zz)

        mse_pde = torch.mean(torch.square(pde))
        return mse_pde , u_z, u

    #Loss of Jacobian-vector product (ensure current step would maximize data variations)
    def loss_jvp(self,mlp,z,t,generator):
        z = z.clone().requires_grad_()
        # z.requires_grad = True
        t = t.clone().requires_grad_()
        # t.requires_grad = True
        u = mlp(z, t)
        u_z = grad(u.sum(), z, create_graph=True)[0]
        z = z + u_z
        u_1 = mlp(z, t + 1)
        u_z1 = grad(u_1.sum(), z, create_graph=True)[0]
        # JVP
        _, jvp_value = jvp(generator, z, v=u_z1, create_graph=True)
        mse_jvp = torch.mean(torch.square(jvp_value))
        return mse_jvp, u_z1


    def forward(self, index, z, t, generator):
        mse_ic = self.loss_ic(self.MLP_SET[index],z)
        mse_pde = 0.0
        half_range = self.num_support_timesteps // 2
        for i in range(0,half_range):
            t_index = i*torch.ones(1,1,requires_grad=True,device=z.device)
            mse_pde_t_index, u_z , u = self.loss_pde(self.MLP_SET[index],z,t_index,self.c[index],generator)
            if i == int(t[0]):
                mse_jvp, u_z1 = self.loss_jvp(self.MLP_SET[index], z, t_index, generator)
                energy = u
                latent1 = z + u_z
                latent2 = z + u_z + u_z1
            z = z + u_z
            mse_pde = mse_pde + mse_pde_t_index
        loss = mse_ic + mse_pde/half_range - mse_jvp
        return energy, latent1, latent2, loss

    def inference(self, index, z, t, generator):
        _, u_z, u = self.loss_pde(self.MLP_SET[index], z, t, self.c[index], generator)
        return u, u_z
