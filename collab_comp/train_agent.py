import os
from argparse import ArgumentParser

from unityagents import UnityEnvironment

from collab_comp.ddpg_algorithm import ddpg
from collab_comp.ddpg_agent import DDPGAgent
from collab_comp.progress_tracker import (
    ScoreGraphPlotter,
    ProgressBarTracker,
)
from collab_comp.solver import AverageScoreOfMaxSolver


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main(output_file, n_rollouts):
    print("Creating Unity environment for Tennis app")
    env = UnityEnvironment(file_name="Tennis.app")

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    brain = env.brains[brain_name]

    print("Initialising PPO Agent")
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    print(f"Using state size {state_size} and action size {action_size}")

    num_agents = len(env_info.agents)
    agent = DDPGAgent(num_agents, state_size, action_size)
    batch_size = 1
    solver = AverageScoreOfMaxSolver(
        solved_score=0.5, solved_score_period=100, num_agents=num_agents
    )
    plotter = ScoreGraphPlotter(
        score_min=0, score_max=0.75, solved_score=0.5, solved_score_period=100
    )
    progress_bar = ProgressBarTracker(n_rollouts)

    ddpg(
        agent,
        num_agents,
        state_size,
        action_size,
        env,
        brain_name,
        n_rollouts,
        batch_size,
        solver,
        output_file,
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
        default=1000,
    )
    args = args_parser.parse_args()

    main(
        args.filename,
        args.n_rollouts,
        args.use_multiple_agents,
        args.algorithm
    )
