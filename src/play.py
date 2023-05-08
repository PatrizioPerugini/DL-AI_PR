from rollout_generator import RolloutGenerator
import torch 
from collections import Counter
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

@dataclass
class HParams():
    
    discrete_action: bool = True
    num_actions: int = 4
    latent_size: int = 32
    hidden_size: int = 256
    action_dim: int = 1

hparams = asdict(HParams())

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rg = RolloutGenerator(hparams,device)
    cumulative_reward = 0
    count_rewards=Counter()
    n_rollouts =10
    for _ in range(n_rollouts):
        rew = - rg.rollout(params=None)
        count_rewards.update([rew])
        cumulative_reward += rew
    print("Reward after {} rollouts: {}".format(n_rollouts, cumulative_reward))
    print("Avg Reward {}".format(cumulative_reward/n_rollouts))

    plt.figure(figsize=(10,10))
    _ = plt.bar(count_rewards.keys(),count_rewards.values()) 
    plt.title("rewards") 
    plt.show()
    print(count_rewards)
   

if __name__ == "__main__":
    main()