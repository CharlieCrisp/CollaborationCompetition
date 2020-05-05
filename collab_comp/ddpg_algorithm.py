from typing import Tuple, List, Union

import numpy as np
import torch

from unityagents import UnityEnvironment, AllBrainInfo

from collab_comp.ddpg_agent import DDPGAgent
from collab_comp.progress_tracker import ProgressTracker
from collab_comp.replay_buffer import PrioritisedReplayBuffer
from collab_comp.solver import Solver

max_t = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(agent: DDPGAgent, target_agent: DDPGAgent, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_agent.actor.parameters(), agent.actor.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    for target_param, local_param in zip(target_agent.critic.parameters(), agent.critic.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def ddpg(
    agents: List[DDPGAgent],
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    update_every: int,
    num_epochs: int,
    learning_epoch_size: int,
    experience_buffer_size: int,
    solver: Solver,
    action_noise_mean,
    action_noise_std,
    priority_eps,
    priority_alpha,
    tau: float = 1e-3,
    progress_trackers: Union[List[ProgressTracker], ProgressTracker] = None,
):
    replay_buffer = PrioritisedReplayBuffer(experience_buffer_size)
    agent1, agent2 = agents
    target_agent1, target_agent2 = agent1.copy(), agent2.copy()

    num_actions = 0
    for i in range(n_rollouts):
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.tensor(env_info.vector_observations).to(device).float()
        score1, score2 = 0, 0
        while not any(env_info.local_done):
            action1 = agent1.act(state[0]).detach().data.cpu().numpy()
            action2 = agent2.act(state[1]).detach().data.cpu().numpy()

            action1 = np.clip(action1 + np.random.normal(action_noise_mean, action_noise_std, 2), -1, 1)
            action2 = np.clip(action2 + np.random.normal(action_noise_mean, action_noise_std, 2), -1, 1)

            actions = np.array([action1, action2])

            env_info = env.step(actions)[brain_name]
            num_actions += 1

            done = env_info.local_done
            next_state = torch.tensor(env_info.vector_observations).to(device).float()

            reward1 = env_info.rewards[0]
            reward2 = env_info.rewards[1]
            score1 += reward1
            score2 += reward2

            priority = (abs(reward1 + reward2) + priority_eps) ** priority_alpha

            replay_buffer.add_experience(
                state,
                torch.tensor(actions),
                torch.tensor(env_info.rewards),
                next_state,
                torch.tensor(done),
                priority,
            )
            state = next_state

        if i % update_every == 0 and len(replay_buffer) > learning_epoch_size:
            for _ in range(num_epochs):
                agent1.learn(target_agent1, replay_buffer, 0)
                agent2.learn(target_agent2, replay_buffer, 1)
                soft_update(agent1, target_agent1, tau)
                soft_update(agent2, target_agent2, tau)

        score = max(score1, score2)
        solver.record_score(score)

        for progress_tracker in progress_trackers:
            progress_tracker.record_score(score)
        if solver.is_solved():
            print(f"Solved in {i} episodes")
            agent1.save_agent_state(1)
            agent2.save_agent_state(2)
