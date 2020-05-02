# Collaboration and Competition
This is a submission for the Udacity Collaboration and Competition Nanodegree project.

## Setting up environment
In order to run this project you will need to setup a conda environment and to download the UnityML environment.
```
conda create -n deep-reinforcement-learning python=3.6
conda activate deep-reinforcement-learning
conda env update -f environment.yml
```

Env downloads for mac:
 - Download [this file](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) and unzip into `Tennis.app`

## Running the project
```
conda activate deep-reinforcement-learning
python -m collab_comp.train_agent -n 750 --use-multiple-agents
```

You use `python -m collab_comp.train_agent --help` to find out more about the CLI interface including how to specify
which UnityML environment to use to train your agent.

## Seeing results
Once you have trained agent weights (or if you want to use the checked in weights) you can see how the agent performs as follows:
```
python -m collab_comp.play_game --use-multiple-agents
```
```
python -m collab_comp.play_game --use-multiple-agents --actor-weights <filepath> --critic-weights <filepath>
```

## Running tests
This repository contains some unit tests. You can run them from the repo root with
 - `conda activate deep-reinforcement-learning`
 - `pytest .`
 
 
## Environment details
In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 
Specifically,

 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
   This yields 2 (potentially different) scores. 
   We then take the maximum of these 2 scores.
 - This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
