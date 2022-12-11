import numpy as np
import torch
from torch import nn


class GradientBandit(nn.Module):
    def __init__(self, n_arms, context_size):
        super(GradientBandit, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(
            in_features=context_size, out_features=n_arms)
        self.n_arms = n_arms
        self.context_size = context_size

    def get_hyperplanes(self):
        weights = torch.cat(
            [self.classifier.weight, self.classifier.bias.reshape(-1, 1)], 1)
        weights = nn.functional.normalize(weights, p=2, dim=1)
        return weights

    def argmax(self, x, agent_rewards):
        '''
        Params
        ------
        x : tensor(n, context_size)
        agent_rewards : tensor(n_arms)
        '''
        x_aug = torch.cat([x, torch.ones(x.shape[0]).reshape(-1, 1)], 1)
        hyperplanes = self.get_hyperplanes()  # (n_arms, context_size + 1)
        # Distances to each hyperplane (n, n_arms)
        distances = torch.abs(x_aug @ hyperplanes.T)
        # Signs of distances to each hyperplane (n, n_arms)
        directions = torch.torch.sign(x_aug @ hyperplanes.T)
        # Willingness to go in the direction of each hyperplane (n, n_arms)
        reward = agent_rewards - distances
        reward_max = torch.max(reward, dim=-1)
        # Which hyperplane would it rather go to? (n)
        indices = reward_max.indices.squeeze()
        # Would the reward be positive? (n)
        positives = (reward_max.values.squeeze() > 0).float()
        # Which one would it rather go to (multiplicable)? (n, n_arms)
        mul_ind = nn.functional.one_hot(indices, num_classes=self.n_arms)
        # Distances/distances to hyperplanes it would rather go to (n, n_arms)
        distances = distances * mul_ind * directions
        # Distances willing to travel to the hyperplanes (n, n_arms)
        distances = (distances.T * positives).T
        # Direction of movement (n, context_size + 1)
        distances = distances @ hyperplanes
        # Moved augmented x values (n, context_size + 1)
        x_aug -= distances
        # The last column (n, d)
        biases = x_aug[:, -1].reshape(-1, 1).tile(1, x_aug.shape[1])
        # Removing last column
        x_prime = (x_aug / biases)[:, :-1]
        return x_prime

    def forward(self, x, agent_rewards):
        '''
        Params
        ------
        x : tensor(n, context_size)
        agent_rewards : tensor(n_arms)

        Returns
        -------
        y_hat : tensor(n, n_arms)
        '''
        x_prime = self.argmax(x, agent_rewards)
        return self.softmax(self.classifier(x_prime))
