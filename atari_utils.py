import gym
import numpy as np


class Squeeze(gym.ObservationWrapper):
    '''Assume wrap_deepmind with scale=True'''

    def __init__(self, env):
        from gym import spaces
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(84, 84), dtype=np.float32)

    def observation(self, observation):
        return np.squeeze(observation)


def make_atari_deepmind(rom_name, valid=False):
    from external.atari_wrappers import make_atari, wrap_deepmind
    env = make_atari(rom_name)
    # framestack is handled by sampler.py
    env = wrap_deepmind(env, episode_life=not valid,
                        frame_stack=False, scale=True)
    env = Squeeze(env)
    return env
