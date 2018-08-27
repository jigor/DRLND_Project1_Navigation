import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling_dqn, dueling_with_learn_eps, fc1_units=137, fc2_units=64, fc1_adv_units=133, fc1_val_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.dueling_dqn = dueling_dqn
        self.act_size = action_size
        self.learn_eps = dueling_with_learn_eps

        if self.dueling_dqn:   # dueling network architecture is selected         
            self.fc1_adv = nn.Linear(fc2_units, fc1_adv_units)
            self.fc1_val = nn.Linear(fc2_units, fc1_val_units)
            if self.learn_eps:
                self.fc1_eps = nn.Linear(fc2_units, 1)
        
            self.fc2_adv = nn.Linear(fc1_adv_units, self.act_size)
            self.fc2_val = nn.Linear(fc1_val_units, 1)
        else:
            self.fc3 = nn.Linear(fc2_units, self.act_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        if self.dueling_dqn:    # dueling network architecture is selected
            adv = F.relu(self.fc1_adv(x))
            val = F.relu(self.fc1_val(x))
            if self.learn_eps:
                eps = F.relu(self.fc1_eps(x))
        
            adv = self.fc2_adv(adv)
            val = self.fc2_val(val).expand(x.size(0), self.act_size)        
            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.act_size)
            if self.learn_eps:
                x = torch.cat((x,eps),1)
        else:
            x = self.fc3(x)

        return x