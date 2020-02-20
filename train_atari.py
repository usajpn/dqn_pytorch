from __future__ import absolute_import
from six.moves import range

import os
import numpy as np

from replay_memory import ReplayMemory
from sampler import Sampler, ObsSampler
from learner import QLearner
from explorer import LinearDecayEGreedyExplorer
from trainer import Trainer
from validator import Validator
from output_path import OutputPath

import torch

from tensorboardX import SummaryWriter


def get_args():

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--gym-env', '-g', default='BreakoutNoFrameskip-v4')
    p.add_argument('--num_epochs', '-E', type=int, default=1000000)
    p.add_argument('--num_episodes', '-T', type=int, default=10)
    p.add_argument('--num_val_episodes', '-V', type=int, default=1)
    p.add_argument('--num_eval_steps', '-S', type=int, default=125000*4)
    p.add_argument('--inter_eval_steps', '-i', type=int, default=250000*4)
    p.add_argument('--num_frames', '-f', type=int, default=4)
    p.add_argument('--render-train', '-r', action='store_true')
    p.add_argument('--render-val', '-v', action='store_true')
    p.add_argument('--extension', '-e', default='cpu')
    p.add_argument('--device-id', '-d', default='0')
    p.add_argument('--log_path', '-l', default='./tmp.output')

    return p.parse_args()


def main():

    args = get_args()

    device = torch.device('cuda', index=args.device_id) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    if args.log_path:
        output_path = OutputPath(args.log_path)
    else:
        output_path = OutputPath()
#    monitor = Monitor(output_path.path)

    tbw = SummaryWriter(output_path.path)

    # Create an atari env.
    from atari_utils import make_atari_deepmind
    env = make_atari_deepmind(args.gym_env, valid=False)
    env_val = make_atari_deepmind(args.gym_env, valid=True)
    print('Observation:', env.observation_space)
    print('Action:', env.action_space)

    # 10000 * 4 frames
    val_replay_memory = ReplayMemory(
        env.observation_space.shape, env.action_space.shape, max_memory=args.num_frames)
    replay_memory = ReplayMemory(
        env.observation_space.shape, env.action_space.shape, max_memory=40000)

    learner = QLearner(env.action_space.n, device, sync_freq=1000, save_freq=250000,
                       gamma=0.99, learning_rate=1e-4, save_path=output_path)

    explorer = LinearDecayEGreedyExplorer(
        env.action_space.n, device, network=learner.get_network(), eps_start=1.0, eps_end=0.01, eps_steps=1e6)

    sampler = Sampler(args.num_frames)
    obs_sampler = ObsSampler(args.num_frames)

    validator = Validator(env_val, val_replay_memory, explorer, obs_sampler,
                          num_episodes=args.num_val_episodes, num_eval_steps=args.num_eval_steps,
                          render=args.render_val, tbw=tbw)

    trainer_with_validator = Trainer(env, replay_memory, learner, sampler, explorer, obs_sampler, inter_eval_steps=args.inter_eval_steps,
                                     num_episodes=args.num_episodes, train_start=10000, batch_size=32,
                                     render=args.render_train, validator=validator, tbw=tbw)

    for e in range(args.num_epochs):
        trainer_with_validator.step()


if __name__ == '__main__':
    main()
