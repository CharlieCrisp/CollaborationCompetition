import os
from argparse import ArgumentParser

from unityagents import UnityEnvironment

from collab_comp.ddpg_algorithm import ddpg
from collab_comp.ddpg_agent import DDPGAgent
from collab_comp.progress_tracker import (
    ScoreGraphPlotter,
    ProgressBarTracker,
)
from collab_comp.solver import AverageScoreSolver


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(n_rollouts):
    print("Creating Unity environment for Tennis app")
    env = UnityEnvironment(file_name="Tennis.app")

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    print("Initialising PPO Agent")
    state_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    num_agents = len(env_info.agents)
    update_every = 1
    tau = 1e-2
    experience_buffer_size = 100000
    num_epochs = 15
    minibatch_size = 128
    actor_lr = 0.001
    critic_lr = 0.001
    action_noise_mean = 0
    action_noise_std = 0.2
    gamma = 0.99
    priority_eps = 0.01
    priority_alpha = 0.5

    agent1 = DDPGAgent(num_agents, state_size, action_size, minibatch_size, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma)
    agent2 = DDPGAgent(num_agents, state_size, action_size, minibatch_size, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma)

    solver = AverageScoreSolver(
        solved_score=0.5, solved_score_period=100, num_agents=num_agents
    )
    plotter = ScoreGraphPlotter(
        score_min=0, score_max=0.75, solved_score=0.5, solved_score_period=100
    )
    progress_bar = ProgressBarTracker(n_rollouts)

    ddpg(
        [agent1, agent2],
        env,
        brain_name,
        n_rollouts,
        update_every,
        num_epochs,
        minibatch_size,
        experience_buffer_size,
        solver,
        action_noise_mean,
        action_noise_std,
        priority_eps,
        priority_alpha,
        tau,
        [plotter, progress_bar],
    )

    print("Finished running PPO. Closing environment")
    env.close()


if __name__ == "__main__":
    args_parser = ArgumentParser(
        description="A script to train and run an agent in the continuous control environment"
    )
    args_parser.add_argument(
        "-n",
        type=int,
        dest="n_rollouts",
        help="The number of trajectories to collect whilst training",
        default=100000,
    )
    args = args_parser.parse_args()

    main(args.n_rollouts)
