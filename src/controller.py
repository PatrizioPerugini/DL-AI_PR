import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass, asdict

@dataclass

class HParams():
    discrete_action: bool = True
    num_actions: int = 4
    latent_size: int = 32
    hidden_size: int = 256
    action_dim: int = 1
    n_workers: int= 4
    pop_size:  int=100
    target_return: int = 0.3
    n_samples: int = 5
    sigma: int = 100
    n_epochs: int = 10000

hparams = asdict(HParams())

class CONTROLLER(pl.LightningModule):
    """ Controller """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        #I might think also of having 4 outputs (1 for each action) and the take the max
        #in this way this should work as a value function
        self.fc = nn.Linear(self.hparams.latent_size + self.hparams.hidden_size, self.hparams.action_dim)

    def forward(self, inputs): 
        #input is a list [latents, recurrents]
        cat_in = torch.cat(inputs, dim=1)
        return torch.tanh(self.fc(cat_in))
        
    def act(self, inputs):
        #0 LEFT
        #1 DOWN
        #2 RIGHT
        #3 UP
        continuous_action = self(inputs)
        if not self.hparams.discrete_action:
            return continuous_action
        #MAP INTO ONLY 2 ACTIONS
        discrete_action = torch.ones_like(continuous_action)
        discrete_action[continuous_action<0]=2
     
        #discrete_action[continuous_action<0.2 ]=3
        #discrete_action[continuous_action<0 ]=2
        #discrete_action[continuous_action<-0.2 ]=0

        return discrete_action
if __name__ == '__main__':
    bs = 4
    z  = torch.rand(bs,32)
    rec = torch.rand(bs,256)
    l = [z,rec]
    m = CONTROLLER(hparams=hparams)
    print(m.forward(l))
    print(m.act(l))     