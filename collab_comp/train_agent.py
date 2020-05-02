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
    batch_size = 2000
    tau = 1e-3
    experience_buffer_size = 10000
    learning_epoch_size = 1000

    agent = DDPGAgent(num_agents, state_size, action_size, learning_epoch_size)
    solver = AverageScoreSolver(
        solved_score=0.5, solved_score_period=100, num_agents=num_agents
    )
    plotter = ScoreGraphPlotter(
        score_min=0, score_max=0.75, solved_score=0.5, solved_score_period=100
    )
    progress_bar = ProgressBarTracker(n_rollouts)

    ddpg(
        agent,
        env,
        brain_name,
        n_rollouts,
        batch_size,
        experience_buffer_size,
        solver,
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
