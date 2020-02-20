from __future__ import absolute_import
from six.moves import range

from collections import namedtuple
import numpy as np

import torch

dtype = torch.float64


from abc import ABC, abstractmethod


class Explorer(ABC):
    def __init__(self, num_actions, device, network=None):
        self.num_actions = num_actions
        self.device = device
        self.network = network

    @abstractmethod
    def select_action(self, obs, val=False):
        raise NotImplementedError()

    def build_network(self, obs):
        pass
#        s = nn.Variable(obs.shape)
#        with nn.parameter_scope(self.name):
#            q = self.q_builder(s, self.num_actions, test=True)
#        Variables = namedtuple(
#            'Variables', ['s', 'q'])
#        self.network = Variables(s, q)

    def build_network_from_nnp(self, nnp_file):
        pass
#        from nnabla.utils import nnp_graph
#        nnp = nnp_graph.NnpLoader(nnp_file)
#        net = nnp.get_network(self.name, batch_size=1)
#        s = net.inputs['s']
#        q = net.outputs['q']
#        Variables = namedtuple(
#            'Variables', ['s', 'q'])
#        assert q.shape[1] == self.num_actions
#        self.network = Variables(s, q)


class GreedyExplorer(Explorer):
    def __init__(self, num_actions, device, use_nnp=False,
                 network=None, nnp_file=None):
        self.num_actions = num_actions
        self.device = device
        self.network = network
        if use_nnp:
            if nnp_file is None:
                return
#            super().build_network_from_nnp(nnp_file)

    def select_action(self, obs, val=False):
        obs = torch.from_numpy(obs).to(dtype).to(self.device)
        with torch.no_grad():
            return self.network(obs).max(1)[1].view(1, 1)


class EGreedyExplorer(Explorer):
    def __init__(self, num_actions, device, epsilon=0.0, use_nnp=False,
                 network=None, nnp_file=None, rng=np.random):
        self.num_actions = num_actions
        self.device = device
        self.epsilon = epsilon
        self.network = network
        self.rng = rng
        if use_nnp:
            if nnp_file is None:
                return
#            super().build_network_from_nnp(nnp_file)

    def select_action(self, obs, val=False):
        obs = torch.from_numpy(obs).to(dtype).to(self.device)
        if self.rng.rand() >= self.epsilon:
            with torch.no_grad():
                return self.network(obs).max(1)[1].view(1, 1)
        return self.rng.randint(self.num_actions, size=(obs.shape[0],))


class LinearDecayEGreedyExplorer(EGreedyExplorer):
    def __init__(self, num_actions, device, eps_val=0.05, eps_start=.9, eps_end=.1, eps_steps=1e6,
                 use_nnp=False, network=None, nnp_file=None,
                 rng=np.random):
        super().__init__(num_actions, device, epsilon=eps_start, use_nnp=use_nnp,
                         network=network, nnp_file=nnp_file, rng=rng)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.eps_val = eps_val
        self.time = 0

    def update(self):
        self.time += 1

    def linear_decay_epsilon(self):
        self.epsilon = max(
           self.eps_end,
           self.eps_start +
           (self.eps_end - self.eps_start) * self.time / self.eps_steps)

    def select_action(self, obs, val=False):
        if val:
            self.epsilon = self.eps_val
        else:
            self.linear_decay_epsilon()
        return super().select_action(obs)
