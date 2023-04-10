import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Tuple, List, Any, Dict, Optional
import os.path
import numpy as np
import torch
from ERB import Experience_replay_buffer
from data import fill_erb
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from VAE import VAE
@dataclass
class HParams():
    # dataset stuff
    batch_size: int = 128
    n_cpu: int = 8 
    lr: int = 1e-3
    wd: int = 1e-5
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    #output_dim: int = 1024
    #dropout: float = 0.3
    latent_size: int = 64
    img_channels : int = 3
    beta: float= 0.01

hparams = asdict(HParams())
class ErbDataset(Dataset):
    def __init__(self,erb ):
        #self.hparams = hparams
        self.data = self.prepare_data(erb)
       
        
    def prepare_data(self,erb:Experience_replay_buffer):
        obs, act, reward, done, next_obs = [], [], [], [], []
        dim = erb.current_dimension
        print(dim)
        for i in range(dim):
            obs.append(erb.replay_memory[i][0])
            act.append(erb.replay_memory[i][1])
            reward.append(erb.replay_memory[i][2])
            done.append(int(erb.replay_memory[i][3]==True))
            next_obs.append(erb.replay_memory[i][4])

        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                    ])
        data = list()
        for k in range(dim):
            item = dict()
            item["obs"] = transform(obs[k]) 
            item["act"] = torch.tensor(act[k])
            item["done"] = torch.tensor(done[k])
            #HERE I SHOULD ALSO ADD THE REWARD AND THE NEXT STATE, BUT FOR THE MOMENT 
            #I DON'T NEED IT TO TRAIN THE VAE
            data.append(item)
       
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleC(pl.LightningDataModule):
    def __init__(self, hparams = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        
    def setup(self, stage: Optional[str] = None):
        map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]
        dataset = fill_erb(map_env, fill_dim=7000)
        data = ErbDataset(dataset)
        split_size=int(len(data)*9/10)
        self.data_train, self.data_test = torch.utils.data.random_split(data, \
                                        [split_size, len(data)-split_size])
        print(len(self.data_train))
        print(len(self.data_test))
        

    def train_dataloader(self):
        #return DataLoader(
        #        self.data_train, 
        #        batch_size = self.hparams.batch_size, 
        #        shuffle = True,
        #        num_workers = self.hparams.n_cpu,
        #        pin_memory=True,
        #        persistent_workers=True
        #    )
        return DataLoader(
                self.data_train, 
                batch_size = 128, 
                shuffle = True,
                num_workers = 8,
                pin_memory=True,
                persistent_workers=True
            )

    def val_dataloader(self):
       
       #return DataLoader(
       #         self.data_test, 
       #         batch_size = self.hparams.batch_size, 
       #         shuffle = False,
       #         num_workers = self.hparams.n_cpu,
       #         pin_memory=True,
       #         persistent_workers=True
       #     )
           return DataLoader(
                self.data_test, 
                batch_size = 128, 
                shuffle = False,
                num_workers = 8,
                pin_memory=True,
                persistent_workers=True
            )

if __name__ == '__main__':
    #TEST DATASET ################################################################
    map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]
    dataset = fill_erb(map_env, fill_dim=100)
    ds = ErbDataset(dataset)
    ##print(ds.__len__()) 
    item = ds.__getitem__(15)
    #item_2 = ds.__getitem__(10)
    #item_3 = ds.__getitem__(3)

   

    ### THIS CODE IS IN ORDER TO CHECK IF THE MODEL HAS LEARNED TO RECONSTRUCT THE EXAMPLES
   
   
    vae = VAE(hparams=hparams)
    x = item['obs'].unsqueeze(0)
    plt.imshow(item['obs'].permute(1,2,0))
    plt.show()
    #print(x.shape)
    x_recon, mu, logsigma, z = vae.forward(x)
    print("reconstruction with random weights")
    plt.imshow(x_recon.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()
    
    model = VAE.load_from_checkpoint("../weights_20k.ckpt")
    x_recon, mu, logsigma, z = model.forward(x)
    print("good model reconstruction")
    plt.imshow(x_recon.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()
    # CHECK DATAMODULE


