from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from collab_comp.policy_nn import Critic, Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(
        self,
        num_agents: int,
        state_size: int,
        action_size: int,
        epsilon=0.2,
        learning_rate=0.0006,
        saved_weights=None,
    ):
        self.action_size = action_size
        self.num_agents = num_agents
        self.epsilon = epsilon

        # TODO - does this require Actor and Critic

        self.actor = Actor(state_size, action_size, seed=0).to(device)
        self.old_actor: nn.Module = Actor(state_size, action_size, seed=0).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        if saved_weights is not None:
            actor_file, critic_file = saved_weights
            self.actor.load_state_dict(torch.load(actor_file))
            self.critic.load_state_dict(torch.load(critic_file))

    def act(self, state: torch.Tensor) -> np.ndarray:
        return self.actor.act(state).detach()

    def _log_probabilities_and_entropies_of_actions(
        self, states: torch.Tensor, actions: torch.Tensor, detach=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param detach: When true, the calculation will be done detached and therefore will not affect auto-diff.
                       The output will be treated as a constant when gradients are calculated.
        :return: The probabilities of taking the given actions from the given states
        """
        probs, entropies = self.actor.log_probability_and_entropy_of_action(
            states, actions
        )

        if detach:
            probs = probs.detach()
            entropies = entropies.detach()
        return probs, entropies

    def learn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        future_rewards,
        num_learning_iterations=80,
    ):
        # TODO complete
        pass

    def save_agent_state(self):
        torch.save(self.actor.state_dict(), "saved_actor.pth")
        torch.save(self.critic.state_dict(), "saved_critic.pth")
