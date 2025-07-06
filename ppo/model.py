import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self, in_channels, out_dim):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.linear1 = nn.Linear(32*9**2, 256)
        self.linear2 = nn.Linear(256, out_dim)

    def forward(self, input):
        output = self.conv1(input)
        output = nn.functional.relu(output)

        output = self.conv2(output)
        output = nn.functional.relu(output)

        output = torch.flatten(output, start_dim=1)

        output = self.linear1(output)
        output = nn.functional.relu(output)

        output = self.linear2(output)
        return output
    

class SmallModel(nn.Module):
    
    def __init__(self, in_channels, out_dim):
        super().__init__()

        self.actor = CNNModel(in_channels, out_dim)
        self.critic = CNNModel(in_channels, 1)

    def forward(self, input):
        actor_output =  self.actor(input)
        actor_output = torch.cat((actor_output, -actor_output), 1)
        return actor_output, self.critic(input)


class BigModel(nn.Module):

    def __init__(self, in_channels, out_dim):

        super().__init__()

        self.base = nn.Sequential(nn.Conv2d(in_channels, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(7*7*64, 512),
                                  nn.ReLU())
        
        self.actor = nn.Sequential(nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, out_dim))
        
        self.critic = nn.Sequential(nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 1))

    def forward(self, input):
        base_output = self.base(input)
        return self.actor(base_output), self.critic(base_output)
