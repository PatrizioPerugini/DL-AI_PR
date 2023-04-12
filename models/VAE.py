import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils
import pytorch_lightning as pl
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping

from dataclasses import dataclass, asdict

@dataclass
class HParams():
    # dataset stuff
    batch_size: int = 128
    n_cpu: int = 8 
    lr: int = 3e-4
    wd: int = 0
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    #output_dim: int = 1024
    dropout: float = 0.3
    latent_size: int = 32
    img_channels : int = 3
    beta: float= 0.4

hparams = asdict(HParams())
#The code is based on the original implementation and is based on the structure explained in the paper
class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, self.img_channels, 6, stride=2)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = torch.sigmoid(self.deconv4(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(2*2*256, self.latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, self.latent_size)

    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma     

class VAE(pl.LightningModule):
    """ Variational Autoencoder """
    def __init__(self, hparams):
        super(VAE, self).__init__()
        self.save_hyperparameters(hparams)
        self.encoder = Encoder(self.hparams.img_channels, self.hparams.latent_size)
        self.decoder = Decoder(self.hparams.img_channels, self.hparams.latent_size)
       
        

    def forward(self, x) :
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        x_recon = self.decoder(z)
        return x_recon, mu, logsigma, z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self,recon_x, x, mu, logsigma):
        """ VAE loss function """
        #original
        BCE = F.mse_loss(recon_x, x, reduction='sum')
 
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        beta = self.hparams.beta
        return {"loss": BCE + beta*KLD, "BCE": BCE, "KLD": KLD}

    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        recon_batch, mu, logvar,_ = self(obs)
        loss = self.loss_function(recon_batch, obs, mu, logvar)
        self.log_dict(loss)
        return loss['loss']

    def validation_step(self, batch, batch_idx) :
        obs = batch['obs']
        recon_batch, mu, logvar, _ = self(obs)
        loss = self.loss_function(recon_batch, obs, mu, logvar)
        images = self.get_image_examples(obs, recon_batch)
        return {"loss_vae_val": loss['loss'], "images": images}
