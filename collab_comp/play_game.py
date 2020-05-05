import os
import numpy as np
import torch

from argparse import ArgumentParser
from unityagents import UnityEnvironment

from collab_comp.ddpg_agent import DDPGAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(n_rollouts):
    print("Creating Unity environment for Tennis app")
    env = UnityEnvironment(file_name="Tennis.app")

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    brain = env.brains[brain_name]

    state_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    num_agents = len(env_info.agents)

    # unused hyperparams
    minibatch_size = 128
    actor_lr = 0.001
    critic_lr = 0.001
    gamma=0.99

    agent1 = DDPGAgent(num_agents, state_size, action_size, minibatch_size, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, saved_weights_agent_no=1)
    agent2 = DDPGAgent(num_agents, state_size, action_size, minibatch_size, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, saved_weights_agent_no=2)

    for i in range(n_rollouts):
        env_info = env.reset(train_mode=False)[brain_name]
        state = torch.tensor(env_info.vector_observations).to(device).float()
        while not any(env_info.local_done):
            action1 = agent1.act(state[0]).detach().data.cpu().numpy()
            action2 = agent2.act(state[1]).detach().data.cpu().numpy()

            actions = np.array([action1, action2])

            env_info = env.step(actions)[brain_name]
            state = torch.tensor(env_info.vector_observations).to(device).float()
            if np.any(env_info.local_done):
                break

    print("Closing environment")
    env.close()


if __name__ == "__main__":
    args_parser = ArgumentParser(
        description="A script to run forward the continuous control environment using a trained agent"
    )
    args_parser.add_argument(
        "-n",
        type=int,
        dest="n_rollouts",
        help="The number of trajectories to collect whilst training",
        default=1000,
    )
    args = args_parser.parse_args()

    main(args.n_rollouts)
