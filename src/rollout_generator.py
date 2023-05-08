
from VAE import VAE
from MDNRNN import MDNRNN
from controller import CONTROLLER
import torch
from torchvision import transforms
import gym
from os.path import exists
from pickle import FALSE
import time
from dataclasses import dataclass, asdict

@dataclass
class HParams():
    
    discrete_action: bool = True
    num_actions: int = 4
    latent_size: int = 32
    hidden_size: int = 256
    action_dim: int = 1

hparams = asdict(HParams())

def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
#taken from the original implementation
def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened
#taken from the original implementation
def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

class RolloutGenerator(object):
    def __init__(self, hparams, device):
        """ Build vae, rnn, controller and environment. """
        
        self.hparams = hparams
        self.device = device
        self.time_limit = 15
        self.visualize = True
        vae_pth = "../weights_continuous.ckpt"
        mdnrnn_pth = "../prova_md.ckpt"
        controller_pth = "../controller_lake.ckpt"
 
        self.vae = VAE.load_from_checkpoint(vae_pth).to(self.device).to(self.device)
        self.mdnrnn = MDNRNN.load_from_checkpoint(mdnrnn_pth, strict = False).to(self.device)
       
        self.controller = CONTROLLER(hparams).to(self.device)
        #map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]
        if exists(controller_pth):
            self.controller.load_state_dict(torch.load(controller_pth, map_location=self.device)['state_dict'])
        #self.env = gym.make("FrozenLake-v1", desc=map_env, render_mode="rgb_array",is_slippery = False)
        self.env = gym.make("FrozenLake-v1", render_mode="rgb_array",is_slippery = False)

    def get_action_and_transition(self, obs, hidden):
        _, _, _, z = self.vae(obs)
        action = self.controller.act([z, hidden[0][0]]).unsqueeze(0)
        (_, _, _, done), next_hidden = self.mdnrnn(z.unsqueeze(0), action.to(self.device), hidden = hidden)
        return action.squeeze().cpu().numpy(), next_hidden, done.view(-1).detach().cpu().numpy()[0]

    def rollout(self, params):
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)
        hy,_ = self.env.reset()
        obs = self.env.render()
        hidden = [torch.zeros(2, 1, 256).to(self.device) for _ in range(2)] 
        transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                    ])
        cumulative = 0
        i = 0
        while True:
            if self.visualize:
                self.env.render()
                time.sleep(0.1)
            
            obs = transform(obs).unsqueeze(0).to(self.device) 
           
            action, hidden, pdone = self.get_action_and_transition(obs, hidden)
            obs = self.env.render()
            s, reward, done, _, _ = self.env.step(int(action))
            cumulative += reward 
            if done or i > self.time_limit: 
                return - cumulative
            i += 1
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    #gpu = p_index % torch.cuda.device_count()
    rg = RolloutGenerator(hparams,device)
   # print(rg.rollout(None))