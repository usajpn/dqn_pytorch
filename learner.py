from __future__ import absolute_import
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple
import os

dtype = torch.float64
torch.set_default_dtype(dtype)

class QCNN(nn.Module):
    def __init__(self, n_actions, state_dim=4):
        super(QCNN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

class QMLP(nn.Module):
    def __init__(self, n_actions):
        super(QMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class QLearner(object):
    def __init__(self, n_actions, device, clip_reward=1.0,
                 sync_freq=500, save_freq=10000, save_path=None,
                 gamma=0.99, learning_rate=5e-4, weight_decay=0, cnn=True):
        self.built = False
        self.n_actions = n_actions
        self.device = device
        self.clip_reward = clip_reward
        self.sync_freq = sync_freq
        self.save_freq = save_freq
        if save_path is None:
            import output_path
            save_path = output_path.default_output_path()
        self.save_path = save_path
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # count of neural network update (steps / train_freq)
        self.update_count = 0

        # build q networks
        if cnn == True:
            self.q = QCNN(n_actions)
            self.q_target = QCNN(n_actions)
        else:
            self.q = QMLP(n_actions)
            self.q_target = QMLP(n_actions)

        self.q.to(device)
        self.q_target.to(device)

        self.optimizer = optim.Adam(self.q.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

    def get_network(self):
        return self.q

    def save_model(self):
        pass

    def update(self, batch):
        s = torch.from_numpy(batch[0]).to(dtype).to(self.device)
        a = torch.from_numpy(batch[1]).long().to(self.device)
        r = torch.from_numpy(batch[2]).long().to(self.device)
        t = torch.from_numpy(batch[3]).long().to(self.device)
        snext = torch.from_numpy(batch[4]).to(dtype).to(self.device)

        # 1. forward-pass

        # reward clipping
        clipped_r = torch.clamp(r, min=-self.clip_reward, max=self.clip_reward)

        # get current q value by taking the q value of the action taken
        q_a = self.q(s).gather(1, a.unsqueeze(1))

        # detach to prohibit backpropagating for target q network
        next_q_max = self.q_target(snext).detach().max(1)[0]
        q_a_target = (clipped_r + self.gamma * (1 - t) * next_q_max).view(-1, 1)

        loss = F.smooth_l1_loss(q_a, q_a_target)

        # 2. update param
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_count % self.sync_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        if self.update_count % self.save_freq == 0:
            self.save_model()

        return loss.clone().data.cpu().numpy()
