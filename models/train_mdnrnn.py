import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from VAE import VAE
from datamodule import Experience_replay_buffer,DataModuleC
from MDNRNN import MDNRNN
# reproducibility stuff
import numpy as np
import random
import torch
from dataclasses import dataclass, asdict

@dataclass
class HParams():
    n_epochs : int = 20
    batch_size: int = 128
    n_cpu: int = 8 
    lr: int = 1e-3
    wd: int = 1e-4
    lstm_num_layers: int = 2
    z_size: int = 32
    n_hidden: int = 256 #(lstm hidden size and lay)
    n_gaussians: int = 5
    seq_len : int = 4 # consider the possibility of giving a contest concatenating more then 1 obs
    action_dim: int = 1

hparams = asdict(HParams())


def train():
    
    dataloader = DataModuleC()
    # Instantiate the model
    mdnrnn = MDNRNN(hparams=hparams)
 
    # Define the trainer
    metric_to_monitor = 'loss'
    early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, patience=15, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
                            save_top_k=1,
                            monitor = metric_to_monitor,
                            mode = "min",
                            filename ="mdnrnn-{epoch:02d}-{avg_val_loss_mdnrnn:.4f}",
                            verbose = True
                        )
    trainer = pl.Trainer(max_epochs=50, 
                        callbacks=[early_stop_callback, checkpoint_callback])   
    
      
    # Start the training
    print("STARTING TRAINING")
    trainer.fit(mdnrnn,dataloader)
    print("END OF TRAINING")
    # Log the trained model
    model_pth ="../prova_md.ckpt"
    trainer.save_checkpoint(model_pth)
   

if __name__ == "__main__":
    train()