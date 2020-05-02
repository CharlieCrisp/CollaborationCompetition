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
    for target_param, local_param in zip(target_agent.optimal_action_picker.parameters(), agent.optimal_action_picker.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    for target_param, local_param in zip(target_agent.action_value_estimator.parameters(), agent.action_value_estimator.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def ddpg(
    agent: DDPGAgent,
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    batch_size: int,
    experience_buffer_size: int,
    solver: Solver,
    tau: float = 1e-3,
    progress_trackers: Union[List[ProgressTracker], ProgressTracker] = None,
):
    replay_buffer = PrioritisedReplayBuffer(experience_buffer_size)
    target_agent = agent.copy()

    num_actions = 0
    for i in range(n_rollouts):
        env_info = env.reset(train_mode=True)[brain_name]
        state = torch.tensor(env_info.vector_observations).to(device).float()
        score1, score2 = 0, 0
        while not any(env_info.local_done):
            state1 = state[0]
            state2 = state[1]
            action1 = agent.act(state1).detach().data.cpu().numpy()
            action2 = agent.act(state2).detach().data.cpu().numpy()

            env_info = env.step(np.array([action1, action2]))[brain_name]
            num_actions += 1

            done = env_info.local_done
            next_state = torch.tensor(env_info.vector_observations).to(device).float()

            reward1 = env_info.rewards[0]
            reward2 = env_info.rewards[1]
            score1 += reward1
            score2 += reward2

            replay_buffer.add_experience(
                state1,
                torch.tensor(action1),
                torch.tensor(reward1),
                next_state[0],
                torch.tensor(done[0]),
                max(replay_buffer.priorities or [0]),
            )

            replay_buffer.add_experience(
                state2,
                torch.tensor(action2),
                torch.tensor(reward2),
                next_state[1],
                torch.tensor(done[1]),
                max(replay_buffer.priorities or [0]),
            )

            if num_actions % batch_size == 0 and num_actions > 0:
                new_priority_indices, new_priorities = agent.learn(target_agent, replay_buffer)
                replay_buffer.update_priority(new_priority_indices, new_priorities.detach().data.cpu().numpy())

            soft_update(agent, target_agent, tau)
            state = next_state

        score = max(score1, score2)
        solver.record_score(score)

        for progress_tracker in progress_trackers:
            progress_tracker.record_score(score)
        if solver.is_solved():
            agent.save_agent_state()
