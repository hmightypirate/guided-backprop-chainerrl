"""An example of training A3C against OpenAI Gym Envs.

This script is an example of training a PCL agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_a3c_gym.py 8 --env CartPole-v0

To solve InvertedPendulum-v1, run:
    python train_a3c_gym.py 8 --env InvertedPendulum-v1 --arch LSTMGaussian --t-max 50  # noqa
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
import numpy as np

from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl import policy
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function
from saliency_a3c import SaliencyA3C

# My imports
from iclr_acer_link import ICLRACERHead, ICLRACERHeadMini
from guided_relu import guided_relu
import env_gym_chainer
import sys

X_SHAPE = 84
Y_SHAPE = 84

def phi(obs):
    assert len(obs) == 4
    raw_values = np.asarray(obs, dtype=np.float32)
    raw_values /= 255.0
    return raw_values


class A3CFF(chainer.ChainList, a3c.A3CModel):
    def __init__(self, n_actions):
        self.head = ICLRACERHead(activation=guided_relu)
        self.pi = policy.FCSoftmaxPolicy(
            self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


def main():
    import logging

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--arch', type=str, default='FFSoftmax',
                        choices=('FFSoftmax', 'FFMellowmax', 'LSTMGaussian'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--frame-buffer-length', type=int, default=4)
    parser.add_argument('--render-b2w', action='store_true', default=False)
    # Additional params
    parser.add_argument('--min_reward', type=float, default=sys.float_info.min)
    
    args = parser.parse_args()

    def f_trim_reward(x, min_reward=args.min_reward):
        if x < min_reward:
            x = 0
        else:
            if x != 0:
                print ("XXXXXXXXXXXXX ", x)
        return x

    logging.getLogger().setLevel(args.logger_level)

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        if args.monitor and process_idx == 0:
            env = gym.wrappers.Monitor(env, args.outdir)
        # Scale rewards observed by agents
        if not test:
            misc.env_modifiers.make_reward_filtered(
                env, lambda x: f_trim_reward(x) * args.reward_scale_factor)
            misc.env_modifiers.make_reward_clipped(env, -1, 1)
        if args.render and process_idx == 0 and not test:
            misc.env_modifiers.make_rendered(env)

        env = env_gym_chainer.GymEnvironment(
            env,
            res_width=X_SHAPE,
            res_height=Y_SHAPE,
            agent_history_length=args.frame_buffer_length,
            render=args.render_b2w)
            
        return env

    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    
    # Switch policy types accordingly to action space types
    model = A3CFF(action_space.n)

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    # Clipping by gradient norm (changed from 40 to 10)
    # opt.add_hook(chainer.optimizer.GradientClipping(10))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = SaliencyA3C(model, opt, t_max=args.t_max, gamma=0.99,
                        beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)

    if args.demo:
        with chainer.using_config("train", False):
            env = make_env(0, True)
            eval_stats = experiments.eval_performance(
                env=env,
                agent=agent,
                n_runs=args.eval_n_runs,
                max_episode_len=timestep_limit)
            print('n_runs: {} mean: {} median: {} stdev {}'.format(
                args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
                eval_stats['stdev']))
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
