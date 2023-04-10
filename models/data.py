import numpy as np
import gym
import math
from statistics import mean
from gym.wrappers.pixel_observation import PixelObservationWrapper as gymWrapper
import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  gym.wrappers.pixel_observation import PixelObservationWrapper #, Grayscale, Resize
from ERB import Experience_replay_buffer
### TEST ONLY
from VAE import VAE

from dataclasses import dataclass, asdict

@dataclass
class HParams():
    # dataset stuff
    batch_size: int = 256
    n_cpu: int = 8 
    lr: int = 3e-4
    wd: int = 0
    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2
    #output_dim: int = 1024
    dropout: float = 0.3
    latent_size: int = 32
    img_channels : int = 3
    beta: float= 0.2

hparams = asdict(HParams())

###

#brownian motion for samoling continuous action space randomly.. more consintent
def sample_continuous_policy(action_space, seq_len, dt):
    actions = [action_space.sample()]
    print(actions)
    for _ in range(seq_len):
        #print(actions[-1])
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions


#returns an experience replay buffer populated as follows:
# (states, actions, rewards, dones, next_states)
# -> change fill_dim in order to insert more examples 
def fill_erb(map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"],fill_dim=1000):
    erb = Experience_replay_buffer()
    #env_render = gym.make("FrozenLake-v1", desc=map_env, render_mode="rgb_array",is_slippery = False)
    env_render =  gym.make('CartPole-v0',render_mode="rgb_array")
    states, actions, rewards, dones, next_states = [],[],[],[],[]
    #for ep in range(num_episodes):
    while erb.current_dimension < fill_dim:  
        done = False
        s, _ = env_render.reset()
        while not done:
            a = env_render.action_space.sample()
            #obs = env.render("rgb_array")
            obs = env_render.render()
            s, r, done, _, _ = env_render.step(a)
            next_obs = env_render.render()
            erb.append(obs,a,r,done,next_obs)

    return erb


if __name__ == '__main__':
    num_episodes = 50
    
    #map_env = generate_random_map(size=6)
    map_env = ["SFFFHF", "FFFFFF", "FHFFFH", "FFFFFF", "HFHFFG"]

    seq_len = 50
    #actions = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

    dataset = fill_erb(map_env, fill_dim=100)
    #print(dataset.replay_memory[7][3])
    #print(int(dataset.replay_memory[7][3]==True))
    batch=dataset.sample_batch()
    states, actions, rewards, dones, next_states= list(batch)
    plt.imshow(states[18])
    plt.show()
    vae = VAE(hparams=hparams)
    model = VAE.load_from_checkpoint("../weights.ckpt")
    plt.imshow(model.forward(states[18]))
    plt.show()
    #print(actions[18])
    #plt.imshow(next_states[18])
    #plt.show()
    #print(dones[18])
    #print(dataset.current_dimension)
 