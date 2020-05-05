# Collaboration and Competition Report
This is the report for my submission for the Udacity Collaboration and Competition Nanodegree project.
Here I will describe the algorithms and details of how I solved this problem.


## Algorithm
This project implements [Deep Deterministic Policy Gradients](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).
This algorithm uses an actor to learn the action that maximises reward and a critic to train the actor against.


More specifically this project implements [Multi Agent Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1706.02275.pdf) where each agent trains
its own policy using local observations, but trains its critic using global observations.


My project also uses a prioritized experience replay which takes the reward plus a baseline priority as the sample priority.

## Details
```python
# hyperparameters
update_every = 1                 # how many episodes to sample before learning
tau = 1e-2                       # soft update factor
experience_buffer_size = 100000  # man number (s, a, r, s, d) samples kept in the replay buffer 
num_epochs = 15                  # how many times to perform learning at any timestep
minibatch_size = 128             # how many data points to sample every epoch
actor_lr = 0.001                 # Adam learning rate of actor
critic_lr = 0.001                # Adam learning rate of critic
action_noise_mean = 0            # The mean of the gaussian noise added to agent actions
action_noise_std = 0.2           # The std of the gaussian noise added to agent actions
gamma = 0.99                     # The discount factor
priority_eps = 0.01              # Additive factor to rewards to ensure non-zero sampling priorities
priority_alpha = 0.5             # Power of rewards when calculating priorities

# Actor
layer1_size = 128
layer2_size = 64
layer3_size = 32

# Critic
layer1_size = 128
layer2_size = 64
layer3_size = 32
```

## Results
My agent was able to solve this environment in 1221 episodes.
![My agent](solved_agent_1340_episodes.png)

## Notes
I have tried using the loss as a priority for experience sampling but found it produced worse performance.
Source code for this can be found on the branch `fix/proper-per`. 

## Future work
Here are my ideas on how I could improbe this project going forward:
 - I could try combining actor and critic into same network so that they share weights.
 This could save training time because the features learnt by each are shared
 - Noise
     - The model currently implements random guassian noise added to the actions.
     - An alternative would be to use [parameter space noise](https://arxiv.org/abs/1706.01905) which adds noise to the weight space
     and generally produces better results.
     - I could also try using different methods for producing noise such as an [OU process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)
     - Lastly I could try annealing noise during the training process so that exploration is greater towards the beginning.
     