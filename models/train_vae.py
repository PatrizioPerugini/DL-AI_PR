
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from VAE import VAE
from datamodule import Experience_replay_buffer,DataModuleC
# reproducibility stuff
import numpy as np
import random
import torch
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False
_ = pl.seed_everything(0)
from dataclasses import dataclass, asdict
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

def train():
   
    #dataloader = WMRLDataModule(hparams = hparams)
    # Instantiate the model
    vae = VAE(hparams=hparams)
    # Define the logger
    dataloader = DataModuleC()
    # Define the trainer
    metric_to_monitor = 'loss'#"loss"
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=15, verbose=True, mode="min")
    #checkpoint_callback = ModelCheckpoint(
    #                        save_top_k=1,
    #                        monitor = metric_to_monitor,
    #                        mode = "min",
    #                        #dirpath = get_env('PTH_FOLDER'),
    #                        filename ="vae-{epoch:02d}-{avg_val_loss_vae:.4f}",
    #                        verbose = True
    #                    )
    trainer = pl.Trainer(max_epochs=300,#hparams.n_epochs, 
                        
                        callbacks=[early_stop_callback])#, checkpoint_callback])    
    
    # Start the training
    trainer.fit(vae,dataloader)
    # Log the trained model
    model_pth = "../weights_20k.ckpt"
    trainer.save_checkpoint(model_pth)
    

if __name__ == "__main__":
    train()
    #vae = VAE(hparams=hparams)
    #model = VAE.load_from_checkpoint("../weights.ckpt")
