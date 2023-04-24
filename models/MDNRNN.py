
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils
import pytorch_lightning as pl
from typing import Sequence, List, Dict, Tuple, Optional, Any, Set, Union, Callable, Mapping
from dataclasses import dataclass, asdict
from VAE import VAE

#NB -> HERE IS NOT NEEDED THE DEFINITION OF IT SINCE 
#      IT WILL BE DEFINED IN THE TRAINER AND PASSED FROM THERE

@dataclass
class HParams():
    img_channels : int = 3
    batch_size: int = 128
    n_cpu: int = 8 
    lr: int = 1e-3
    wd: int = 0
    lstm_num_layers: int = 2
    z_size: int = 32
    n_hidden: int = 256 #(lstm hidden size and lay)
    n_gaussians: int = 5
    seq_len : int = 4 # consider the possibility of giving a contest concatenating more then 1 obs 
    action_dim: int = 1

hparams = asdict(HParams())

class MDNRNN(pl.LightningModule):
    def __init__(self,hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        pth =  "../weights_new_f.ckpt"
        self.vae = VAE.load_from_checkpoint(pth)
        self.vae.freeze() # already trained
        self.z_size = self.hparams.z_size
        self.n_gaussians = self.hparams.n_gaussians
        self.n_hidden = self.hparams.n_hidden
        self.batch_size = self.hparams.batch_size
        self.lstm_num_layers = self.hparams.lstm_num_layers
        self.action_dim = self.hparams.action_dim
        self.lstm = nn.LSTM(self.z_size+self.action_dim,
                            self.n_hidden,
                            self.lstm_num_layers,
                            batch_first = True)
        #now we need to compute three parameters (pi - sigma - mu) for each gaussian
        self.pi = nn.Linear(self.n_hidden,self.n_gaussians*self.z_size)      
        self.mu = nn.Linear(self.n_hidden,self.n_gaussians*self.z_size)      
        self.sigma = nn.Linear(self.n_hidden,self.n_gaussians*self.z_size)   

        self.d = nn.Linear(self.n_hidden,1)
        #adding now 
       # self.hidden = None
        
        #self.hidden = [torch.zeros(self.hparams.lstm_num_layers, self.batch_size, self.hparams.n_hidden).cuda() for _ in range(2)]
    

    
    def get_mixture_coeffs(self,y):
        rollout_len = y.size(1)
        pi = self.pi(y)
        mu = self.mu(y)
        sigma = self.sigma(y)
        pi = pi.view(-1,rollout_len,self.n_gaussians,self.z_size)
        mu = mu.view(-1,rollout_len,self.n_gaussians,self.z_size)
        sigma = sigma.view(-1, rollout_len, self.n_gaussians, self.z_size)
        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        
        return pi, mu, sigma
        

    def forward(self, z,a,hidden = None) :
        if len(a.shape) ==2:
            a = a.unsqueeze(-1)
        #print("THE SHAPE OF Z IS", z.shape)
        #shape z is (bs,seq_len,hidden_dim)
        if len(z.shape) ==2:
            z = z.unsqueeze(1) #adding seq len = 1
        inp = torch.cat([a,z],dim = -1)#check
        if hidden == None:
            y, (h, c) = self.lstm(inp)
        else:
            y, (h, c) = self.lstm(inp,hidden)    
        
        pi, mu, sigma = self.get_mixture_coeffs(y)
        done = torch.sigmoid(self.d(y))
        return (pi, mu, sigma, done), (h,c)

    #mixture loss
    def mdn_loss_fn(self,y, pi, mu, sigma):
        y = y.unsqueeze(2)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        loss = torch.exp(m.log_prob(y))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss)
        l = loss.mean()
        return l

    #termination loss
    def d_loss_fn(self,out,inp):
        out = out.view(-1,out.shape[-1])
        inp = inp.view(-1,inp.shape[-1])
        loss = F.mse_loss(out,inp)
        return loss

    #get loss .. see if balancing is good
    def loss_function(self,z_next,mu,sigma,pi,pred_d,d):
        mdn_loss = self.mdn_loss_fn(z_next,pi,mu,sigma)
        d_loss   = self.d_loss_fn(pred_d,d)
        bal = 1

        return {'loss': mdn_loss+bal*d_loss, 'MDN_loss': mdn_loss, "D_loss":d_loss}

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }
    
   
    def get_latent(self, obs):
        sl = 1
        bs = obs.shape[0]
        if len(obs.shape)==4:
            sl = 1
        else:
            sl = obs.shape[1] # to support multi framology
        #(bs, sl,num_channel,height,width)
        img_shape = obs.shape[2:]
        obs = obs.reshape(-1,*img_shape)
        #print(obs.shape)
        with torch.no_grad():
            _, mu, logsigma, z = self.vae(obs)
               #z_shape [bs,seq_len,latent_size]
       
        return z.view(bs,sl, self.hparams.z_size)


    #HERE I NEED TO CHANGE SOMETHING SO THAT IT SUPPORTS MULTIPLE IMAGES
    def training_step(self, batch, batch_idx):
        obs = batch['obs']
        #([52, 3, 64, 64
        #print("THE SHAPE OF OBS IS", obs.shape)
        act = batch['act']#.unsqueeze(-1)
        done = batch['done']#.unsqueeze(-1)
        if len(done.shape)==2:
            done = done.unsqueeze(-1)
            act = act.unsqueeze(-1)
        next_obs = batch['next_obs']
        latent_obs = self.get_latent(obs)
        next_latent_obs = self.get_latent(next_obs)
        (pi, mu, sigma, pdone), (_,_) = self(latent_obs,act)
        loss = self.loss_function(next_latent_obs, mu, sigma, pi, pdone, done)
        self.log_dict(loss)
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        obs = batch['obs']
        act = batch['act']#.unsqueeze(-1)
        done = batch['done']#.unsqueeze(-1)
        if len(done.shape)==2:
            done = done.unsqueeze(-1)
            act = act.unsqueeze(-1)
        next_obs = batch['next_obs']
       
        latent_obs = self.get_latent(obs)
        next_latent_obs = self.get_latent(next_obs)
        (pi, mu, sigma, pdone), (_,_) = self(latent_obs,act)
        loss = self.loss_function(next_latent_obs, mu, sigma, pi, pdone, done)
        return {"loss_mdnrnn_val": loss['loss']}

if __name__ == '__main__':
    
    model = MDNRNN(hparams=hparams)
    a = torch.rand(1).unsqueeze(0)
   
    a_s = torch.vstack([a,a]).unsqueeze(0)
  
    z = torch.rand(32).unsqueeze(0)
    z_s = torch.vstack([z,z]).unsqueeze(0)

    (pi, mu, sigma, done), (h,c) = model.forward(a_s,z_s)
    print("done")