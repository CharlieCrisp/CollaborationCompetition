import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss

from collab_comp.policy_nn import ValueEstimatorNN, ActorNN
from collab_comp.replay_buffer import PrioritisedReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        minibatch_size: int,
        actor_lr,
        critic_lr,
        epsilon=0.2,
        gamma=0.99,
        saved_weights=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.minibatch_size = minibatch_size
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.action_value_estimator = ValueEstimatorNN(state_size, action_size).to(device)
        self.optimal_action_picker = ActorNN(state_size, action_size, seed=0).to(device)

        self.action_value_estimator_optimizer = optim.Adam(self.action_value_estimator.parameters(), lr=critic_lr)
        self.optimal_action_picker_optimizer = optim.Adam(self.optimal_action_picker.parameters(), lr=actor_lr)

        if saved_weights is not None:
            action_value_estimator_file, optimal_action_picker_file = saved_weights
            self.action_value_estimator.load_state_dict(torch.load(action_value_estimator_file))
            self.optimal_action_picker.load_state_dict(torch.load(optimal_action_picker_file))

    def act(self, state: torch.Tensor):
        return self.optimal_action_picker(state)

    def learn(self, target_agent, experience_replay_buffer: PrioritisedReplayBuffer):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            sampled_indices,
        ) = experience_replay_buffer.sample(self.minibatch_size)

        # Value estimation update
        q_estimate = self.action_value_estimator(states, actions)

        target_next_actions = target_agent.optimal_action_picker(next_states).detach()
        next_state_value = target_agent.action_value_estimator(next_states, target_next_actions).detach()
        q_target = rewards + self.gamma * (torch.tensor(1) - dones.float()) * next_state_value

        value_loss = (q_estimate - q_target).pow(2)

        self.action_value_estimator_optimizer.zero_grad()
        value_loss.mean().backward()
        self.action_value_estimator_optimizer.step()

        # Policy update
        optimal_actions = self.optimal_action_picker(states)
        self.action_value_estimator.eval()

        q_estimate_of_optimal_actions = self.action_value_estimator(states, optimal_actions)

        action_loss = -q_estimate_of_optimal_actions

        self.optimal_action_picker_optimizer.zero_grad()
        action_loss.mean().backward()
        self.optimal_action_picker_optimizer.step()

        self.action_value_estimator.train()

        return sampled_indices, value_loss

    def save_agent_state(self):
        torch.save(self.action_value_estimator.state_dict(), "saved_value_estimator.pth")
        torch.save(self.optimal_action_picker.state_dict(), "saved_actor.pth")

    def copy(self):
        new_agent = DDPGAgent(
            self.num_agents,
            self.state_size,
            self.action_size,
            self.minibatch_size,
            self.actor_lr,
            self.critic_lr,
            self.epsilon,
            self.gamma
        )
        new_agent.action_value_estimator.load_state_dict(self.action_value_estimator.state_dict())
        new_agent.optimal_action_picker.load_state_dict(self.optimal_action_picker.state_dict())
        return new_agent
