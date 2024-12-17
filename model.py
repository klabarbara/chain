import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_nodes, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_nodes, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_nodes)

    def forward(self, x):
        # x will be a batch of states, each state is a single integer node index
        # Convert integer node index to one-hot
        # If x shape: (batch,), do one-hot
        batch_size = x.shape[0]
        one_hot = F.one_hot(x, num_classes=self.fc3.out_features).float()
        out = F.relu(self.fc1(one_hot))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
