import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss

from collab_comp.policy_nn import Critic, ActorNN
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
        saved_weights_agent_no=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.minibatch_size = minibatch_size
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.critic = Critic(state_size, action_size).to(device)
        self.actor = ActorNN(state_size, action_size, seed=0).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        if saved_weights_agent_no is not None:
            print(f"saved_value_estimator_{saved_weights_agent_no}.pth")
            self.critic.load_state_dict(torch.load(f"saved_critic_{saved_weights_agent_no}.pth"))
            self.actor.load_state_dict(torch.load(f"saved_actor_{saved_weights_agent_no}.pth"))

    def act(self, state: torch.Tensor):
        return self.actor(state)

    def learn(self, target_agent, experience_replay_buffer: PrioritisedReplayBuffer, agent_no):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = experience_replay_buffer.sample(self.minibatch_size)

        # Value estimation update
        q_estimate = self.critic(states, actions[:, agent_no, :])
        target_next_actions = target_agent.actor(next_states[:, agent_no, :])
        next_state_value = target_agent.critic(next_states, target_next_actions)
        q_target = rewards[:, agent_no] + self.gamma * (torch.tensor(1) - dones[:, agent_no].float()) * next_state_value

        value_loss = mse_loss(q_estimate, q_target)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # Policy update
        optimal_actions = self.actor(states[:, agent_no, :])

        q_estimate_of_optimal_actions = self.critic(states, optimal_actions)

        action_loss = -q_estimate_of_optimal_actions

        self.actor_optimizer.zero_grad()
        action_loss.mean().backward()
        self.actor_optimizer.step()

    def save_agent_state(self, agent_no):
        torch.save(self.critic.state_dict(), f"saved_critic_{agent_no}.pth")
        torch.save(self.actor.state_dict(), f"saved_actor_{agent_no}.pth")

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
        new_agent.critic.load_state_dict(self.critic.state_dict())
        new_agent.actor.load_state_dict(self.actor.state_dict())
        return new_agent
