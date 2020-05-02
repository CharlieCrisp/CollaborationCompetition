from typing import Tuple, List, Union

import numpy as np
import torch

from unityagents import UnityEnvironment, AllBrainInfo

from collab_comp.ddpg_agent import DDPGAgent
from collab_comp.progress_tracker import ProgressTracker
from collab_comp.solver import Solver

max_t = 10000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ddpg(
    agent: DDPGAgent,
    num_agents: int,
    state_size,
    action_size,
    env: UnityEnvironment,
    brain_name: str,
    n_rollouts: int,
    batch_size: int,
    solver: Solver,
    solved_agent_output_file: str,
    progress_trackers: Union[List[ProgressTracker], ProgressTracker] = None,
):
    # TODO
    pass
