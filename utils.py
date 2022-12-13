import numpy as np
import torch
from torch import nn


class GradientBandit(nn.Module):
    ''' Gradient-optimizable strategy-aware contextual bandits
    '''

    def __init__(self, n_arms, context_size, bias=True):
        super(GradientBandit, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Linear(
            in_features=context_size, out_features=n_arms, bias=bias)
        self.n_arms = n_arms
        self.context_size = context_size
        self.criterion = nn.MSELoss()
        self.bias = bias

    def get_hyperplanes(self):
        '''
        Returns
        -------
        weights : tensor(n_arms, context_size + 1)
        '''
        if self.bias:
            weights = torch.cat(
                [self.classifier.weight, self.classifier.bias.reshape(-1, 1)], 1).detach()
            weights = nn.functional.normalize(weights, p=2, dim=1)
        else:
            weights = self.classifier.weight.detach()
        # weights = nn.functional.normalize(weights, p=2, dim=1)
        return weights

    def slow_argmax(self, x, agent_rewards, variances=None):
        ''' Correct implementation of argmax (...) when there's no bias!
        '''
        assert not self.bias
        hyperplanes = self.get_hyperplanes().float()
        x_prime = torch.clone(x.detach())
        for i, xi in enumerate(x):
            if not variances is None:
                hyperplanes += torch.tensor(np.random.normal(
                    scale=np.sqrt(variances[i].item()), size=self.n_arms))
            y_hat = torch.max(xi @ hyperplanes.T).indices.item()
            rewards = agent_rewards - agent_rewards[y_hat]
            betas = nn.functional.normalize(
                hyperplanes - hyperplanes[y_hat], p=2, dim=-1)
            distances = xi @ betas.T
            zero_dist = torch.linalg.norm(xi).item()
            zero_y = torch.max((-0.0001 * xi) @ hyperplanes.T).indices.item()
            benefits = rewards - torch.abs(distances)
            zero_benefit = agent_rewards[zero_y] - \
                agent_rewards[y_hat] - zero_dist
            if zero_benefit > torch.max(benefits).values.item():
                x_prime[i] = -0.0001 * xi
            else:
                best = torch.max(benefits).indices.item()
                x_prime[i] = xi + betas[best] * distances[best] * 1.0001
        return x_prime

    def argmax(self, x, agent_rewards, variances=None):
        ''' argmax_{x' \in X} [(r^A(f_\theta (x')) - c(x, x'))]
        Returns the adversarial contexts x'.  
        This assumes that c is the Euclidean distance function.

        Params
        ------
        x : tensor(n, context_size)
            Original contexts.
        agent_rewards : tensor(n_arms)
            Rewards that agents get from each arm being pulled.
        variances : tensor(context_size + 1)
            Variances on noised perception of the classifiers.

        Returns
        -------
        x_prime : tensor(n, context_size)
            Adversarial contexts.
        '''
        x_aug = torch.cat(
            [x, torch.ones(x.shape[0]).reshape(-1, 1)], 1).float()
        hyperplanes = self.get_hyperplanes().float()  # (n_arms, context_size + 1)
        # Noise the perceptions (n_arms, context_size + 1)
        if not variances is None:
            hyperplanes += torch.tensor(np.random.normal(
                scale=np.sqrt(variances.numpy())).reshape(-1, 1))
        # Distances to each hyperplane (n, n_arms)
        distances = torch.abs(x_aug @ torch.t(hyperplanes))
        # Signs of distances to each hyperplane (n, n_arms)
        directions = torch.sign(x_aug @ torch.t(hyperplanes))
        # Willingness to go in the direction of each hyperplane (n, n_arms)
        reward = directions * agent_rewards - distances
        reward_max = torch.max(reward, dim=-1)
        # Which hyperplane would it rather go to? (n)
        indices = reward_max.indices.squeeze()
        # Would the reward be positive? (n)
        positives = (reward_max.values.squeeze() > 0).float()
        # Which one would it rather go to (multiplicable)? (n, n_arms)
        mul_ind = nn.functional.one_hot(
            indices, num_classes=self.n_arms).float()
        # Distances/distances to hyperplanes it would rather go to (n, n_arms)
        distances = distances * mul_ind * directions
        # Distances willing to travel to the hyperplanes (n, n_arms)
        distances = torch.t(torch.t(distances) * positives)
        # Direction of movement (n, context_size + 1)
        distances = distances @ hyperplanes
        # Moved augmented x values (n, context_size + 1)
        x_aug -= distances
        # The last column (n, d)
        biases = x_aug[:, -1].reshape(-1, 1).tile(1, x_aug.shape[1])
        # Removing last column
        x_prime = (x_aug / biases)[:, :-1]
        return x_prime

    def forward(self, x, agent_rewards, variances=None, y=None):
        '''
        Params
        ------
        x : tensor(n, context_size)
        agent_rewards : tensor(n_arms)

        Returns
        -------
        y_hat : tensor(n, n_arms)
        '''
        x_prime = self.slow_argmax(x, agent_rewards, variances=variances)
        y_hat = self.sigmoid(self.classifier(x_prime))
        if y is None:
            return y_hat
        else:
            return self.criterion(y, y_hat)


class Agents(torch.utils.data.Dataset):
    def __init__(self, n, n_arms, context_size,
                 arms=None,
                 max_reward=0.0, max_variance=0.0,
                 bias=True):
        '''
        Nonstrategic agents: max_reward = 0
        Strategic agents: max_reward > 0, max_variance = 0
        Imperfect strategic agents: max_reward > 0, max_variance > 0
        '''
        if arms is None:
            raise NotImplementedError('Too much effort')
        # assert arms.shape == (n_arms, context_size + 1)
        self.n = n
        self.n_arms = n_arms
        self.context_size = context_size
        self.arms = arms
        self.rewards = torch.tensor(np.random.rand(n_arms)) * max_reward
        self.variances = torch.tensor(np.random.rand(n)) * max_variance
        self.x = torch.tensor(np.random.rand(n, context_size))
        if bias:
            x_aug = torch.cat(
                [self.x, torch.ones(n).reshape(-1, 1)], 1).float()
        else:
            x_aug = self.x.float()
        self.y = torch.sign(x_aug @ arms.T)
        self.indices = np.arange(n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return (
            self.x[self.indices[i]],
            self.y[self.indices[i]],
            self.variances[self.indices[i]])
