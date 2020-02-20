# Deep Q Network

This is a PyTorch implementation of DQN.
[Here](https://github.com/sony/nnabla-examples/tree/master/reinforcement_learning/dqn) is the similar implementation in SONY NNabla.

* [Mnih et.al., Human-level control through deep reinforcement learning.](https://www.nature.com/articles/nature14236)

##  Requirements

* gym
```
pip install gym
pip install gym[atari]
```

* TensorboardX
```
pip install tensorboardX
```

## Run

### Training

Run (see options by `-h`):

```
python train_atari.py
```

This will output the following files in a log folder (`.tmp.monitor` by default):

* Mean episodic rewards in each epoch over time.
* Learned models with `nnp` format.


### Playing learned model

Run (see options by `-h`):

```
python play_atari.py
```

### Training parameter difference between ours and OpenAI baseline

This implementation uses [OpenAI baseline](https://github.com/openai/baselines)'s parameter by default.


|                             |  Ours | Baseline (defaults.py)  |
| ----                        |  ---- | ----                    |
| Learning Rate               | 1e-4  | 1e-4                    |
| Start Epsilon               | 1.0   | 1.0                     |
| Final Epsilon               | 0.01  | 0.01                    |
| Epsilon Decay Steps         | 1e6   | 1e6 (when max_step:1e8) |
| Dicount Factor (gamma)      | 0.99  | 0.99                    |
| Batch Size                  | 32    | 32                      |
| Replay Buffer Size          | 10000 | 10000                   |
| Target Network Update Freq. | 1000  | 1000                    |
| Learning Start Step         | 10000 | 10000                   |


## Atari Evaluation

There are 2 steps when evaluating Atari games.

1. Intermediate evaluation during training

    * In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep.
    * The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
    * epsilon is set to 0.05 (not greedy).

2. Final evaluation for reporting

    * The Q-Network parameter (.nnp file) with the best mean reward is used. You can just look through `Eval-Score.series.txt` and find the maximum scored steps. Note that `qnet_XXX.nnp`'s `XXX` represents **steps** and not **frames**, whereas the score output in `Eval-Score.series.txt` are **frames**.

    * Using the best .nnp file, by running as shown below, the specified game is played for 30 episodes with a termination threshold of 4500 steps.

        Run (see options by `-h`):
        ```
        python eval_atari.py
        ```



