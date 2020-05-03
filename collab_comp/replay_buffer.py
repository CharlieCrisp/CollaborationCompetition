import torch
import numpy as np

from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimpleReplayBuffer:
    def __init__(self, maxlen):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)

    def add_experience(self, state, action, reward, next_state, is_done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(is_done)

    def sample(self, num_samples):
        indexes = np.arange(len(self.states))
        sampled_indexes = np.random.choice(indexes, num_samples)

        states = torch.stack([self.states[i] for i in sampled_indexes]).to(device)
        actions = torch.stack([self.actions[i] for i in sampled_indexes]).to(device)
        rewards = torch.stack([self.rewards[i] for i in sampled_indexes]).to(device)
        next_states = torch.stack([self.next_states[i] for i in sampled_indexes]).to(
            device
        )
        dones = torch.stack([self.dones[i] for i in sampled_indexes]).to(device)

        return states, actions, rewards, next_states, dones


class PrioritisedReplayBuffer:
    def __init__(self, maxlen):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.e = 1e-6

    def add_experience(self, state, action, reward, next_state, is_done, priority):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(is_done)
        self.priorities.append(abs(priority) + self.e)

    def sample(self, num_samples):
        indexes = np.arange(len(self.states))
        priorities = np.array(self.priorities)
        p = priorities / sum(priorities)
        sampled_indexes = np.random.choice(indexes, num_samples, replace=False, p=p)

        states = torch.stack([self.states[i] for i in sampled_indexes]).to(device)
        actions = torch.stack([self.actions[i] for i in sampled_indexes]).to(device)
        rewards = torch.stack([self.rewards[i] for i in sampled_indexes]).to(device)
        next_states = torch.stack([self.next_states[i] for i in sampled_indexes]).to(
            device
        )
        dones = torch.stack([self.dones[i] for i in sampled_indexes]).to(device)

        return states, actions, rewards, next_states, dones, sampled_indexes

    def update_priority(self, sampled_indexes, new_priorities):
        for idx, new_priority in zip(sampled_indexes, new_priorities):
            self.priorities[idx] = abs(new_priority) + self.e

    def __len__(self):
        return len(self.states)
