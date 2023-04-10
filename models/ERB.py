import numpy as np
import random
from collections import deque , namedtuple

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

class Experience_replay_buffer:

    def __init__(self, memory_size=50000, burn_in=5000): #change the butn in capacity
        self.memory_size = memory_size
        self.current_dimension = 0
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)
        

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, s_0, a, r, d, s_1):
        self.replay_memory.append(
            self.Buffer(s_0, a, r, d, s_1))
        self.current_dimension+=1
        self.current_dimension = self.current_dimension%self.memory_size

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size