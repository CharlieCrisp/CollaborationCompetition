import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(ActorNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer1_size = 256
        layer2_size = 128
        self.layer1 = nn.Linear(state_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, action_size)

    def forward(self, state):
        x = torch.tanh(self.layer1(state))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x


class ValueEstimatorNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ValueEstimatorNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, state, action):
        input_tensor = torch.cat([state, action], dim=1)
        return self.net(input_tensor).squeeze()
