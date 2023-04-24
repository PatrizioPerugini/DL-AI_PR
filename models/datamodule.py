import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from typing import Tuple, List, Any, Dict, Optional
import os.path
import numpy as np
import torch
from ERB import Experience_replay_buffer
from data import fill_erb
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from VAE import VAE
from MDNRNN import MDNRNN
@dataclass
class HParams():
    # dataset stuff
    img_channels : int = 3
    batch_size: int = 128
    n_cpu: int = 8 
    lr: int = 1e-3
    wd: int = 1e-5
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    latent_size: int = 32
    
    beta: float= 0.01
    z_size: int = 32
    n_hidden: int = 256 #(lstm hidden size and lay)
    n_gaussians: int = 5
    seq_len : int = 1 # consider the possibility of giving a contest concatenating more then 1 obs 
    action_dim: int = 1
hparams = asdict(HParams())

class ErbDataset(Dataset):
    def __init__(self,erb ):
        #self.hparams = hparams
        self.data = self.prepare_data(erb)
       
        
    def prepare_data(self,erb:Experience_replay_buffer):
        obs, act, reward, done, next_obs = [], [], [], [], []
        dim = erb.current_dimension
        seq_l = hparams['seq_len'] +1
        #seq_l = hparams.seq_len +1 #should be correct
        print(dim)
        for i in range(dim):
            obs.append(erb.replay_memory[i][0])
            act.append(erb.replay_memory[i][1])
            reward.append(erb.replay_memory[i][2])
            done.append(float(erb.replay_memory[i][3]==True))
            next_obs.append(erb.replay_memory[i][4])
       
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                    ])
        data = list()
        if seq_l <=2 : #just as before the change
            for k in range(dim):
                item = dict()
                item["obs"] = transform(obs[k]) #shape (3,64,64)
                item["act"] = torch.tensor(act[k])
                item["done"] = torch.tensor(done[k])
                item["reward"] =  torch.tensor(reward[k])
                item["next_obs"] = transform(next_obs[k])
                #HERE I SHOULD ALSO ADD THE REWARD AND THE NEXT STATE, BUT FOR THE MOMENT 
                #I DON'T NEED IT TO TRAIN THE VAE
                data.append(item)
        else:
            for k in range(dim-seq_l):
                item = dict()
                item["obs"] = torch.stack([transform(obs[i]) for i in range(k,k+seq_l)])
                item["act"] = torch.tensor([[act[i]] for i in range(k,k+seq_l)])
                item["done"] = torch.tensor([[done[i]] for i in range(k,k+seq_l)])
                item["rewards"] = torch.tensor([[done[i]] for i in range(k,k+seq_l)])
                item["next_obs"] = torch.stack([transform(next_obs[i]) for i in range(k,k+seq_l)])
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
        #HERE I PUT 100 INSTEAD OF 7K JUST FOR TESTING
        dataset = fill_erb(map_env, fill_dim=5000)
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
    map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]
    dataset = fill_erb(map_env, fill_dim=100)
    ds = ErbDataset(dataset)
    
    item_1 = ds.__getitem__(15)

    item_2 = ds.__getitem__(23)
 
    #VAE
    vae = VAE(hparams=hparams)
    model_pth = "../weights_new_f.ckpt"
    #model = VAE.load_from_checkpoint("../weights_car.ckpt")
    model = VAE.load_from_checkpoint(model_pth)

    x_1 = item_1['obs'].unsqueeze(0)
    #x_1 = item_1['obs'][0].unsqueeze(0)
    next_x1 = item_1['next_obs'].unsqueeze(0)
    print("the shape of x1 is", x_1.shape)
    a1 = item_1['act'].unsqueeze(0)
    x_2 = item_2['obs'].unsqueeze(0)
    x_recon_1, mu_1, logsigma_1, z_1 = model.forward(x_1)
    x_recon_2, mu_2, logsigma_2, z_2 = model.forward(x_2)
   # Grid = make_grid([x_1.squeeze(0), x_recon_1.squeeze(0), x_2.squeeze(0), x_recon_2.squeeze(0)])
    # display result
    #img = torchvision.transforms.ToPILImage()(Grid)
    #img.show()

    #MDNRNN PROOF OF CONCEPT

    mdn = MDNRNN(hparams=hparams)
    model_path = "../prova_md.ckpt"
    model_m = MDNRNN.load_from_checkpoint(model_path)
    #model_m  = MDNRNN(hparams=hparams)

    a_1 = a1.unsqueeze(-1)
    #print(z_1.shape)
    #print(a_1.shape)
    
    (pi, mu, sigma, done), (h,c) = model_m.forward(z_1,a_1)

    y_preds = [torch.normal(mu, sigma)[:, :, i, :] for i in range(5)]

    z_in_0 = y_preds[0].squeeze(0).squeeze(0)
    z_in_1 = y_preds[1].squeeze(0).squeeze(0)
    z_in_2 = y_preds[2].squeeze(0).squeeze(0)
    z_in_3 = y_preds[3].squeeze(0).squeeze(0)
    z_in_4 = y_preds[4].squeeze(0).squeeze(0)
    out_image_0 = model.decode(z_in_0)
    out_image_1 = model.decode(z_in_1)
    out_image_2 = model.decode(z_in_2)
    out_image_3 = model.decode(z_in_3)
    out_image_4 = model.decode(z_in_4)
    print("original image")
    plt.imshow(item_1['obs'].permute(1,2,0))
    plt.show()
    
    print("original NEXT image")
    plt.imshow(item_1['next_obs'].permute(1,2,0))
    plt.show()
    
    print("GUESSESS OF THE MODEL")
    plt.imshow(out_image_0.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()

    plt.imshow(out_image_1.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()

    plt.imshow(out_image_2.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()

    plt.imshow(out_image_3.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()

    plt.imshow(out_image_4.squeeze(0).permute(1,2,0).detach().numpy())
    plt.show()
    #Grid = make_grid([out_image_0.squeeze(0), out_image_1.squeeze(0)])#, out_image.squeeze(0), out_image.squeeze(0)])
    ## display result
    #img = torchvision.transforms.ToPILImage()(Grid)
    #img.show()