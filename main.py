from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import atari_env
from utils import read_config
from model import A3Clstm
from train import train
from test import test
from scheduler import scheduler
from shared_optim import SharedRMSprop, SharedAdam
import sys
#from gym.configuration import undo_logger_setup
import time

import mpi

#undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='G',
    help='discount factor for rewards (default: 0.99)')
parser.add_argument(
    '--report-hours',
    default=False,
    action='store_true',
    help='record results by hours instead of eps')
parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')
parser.add_argument(
    '--log_interval',
    type=int,
    default=60,
    metavar='L',
    help='interval between training status logs (seconds)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--min_steps',
    type=int,
    default=20,
    help='lower bound for offline steps (default: 20)')
parser.add_argument(
    '--max_steps',
    type=int,
    default=150,
    help='upper bound for offline steps (default: 150)')
parser.add_argument(
    '--steps_mean',
    default=150,
    type=float,
    help='normal mean for offline sampling')
parser.add_argument(
    '--min_delay',
    type=int,
    default=20,
    help='lower bound for offline delay (default: 20)')
parser.add_argument(
    '--max_delay',
    type=int,
    default=150,
    help='upper bound for offline delay (default: 150)')
parser.add_argument(
    '--max_time',
    type=int,
    default=100000,
    help='total federation time steps')
parser.add_argument(
    '--delay_mean',
    default=150,
    type=float,
    help='normal mean for offline sampling')
parser.add_argument(
    '--delay_var',
    default=25,
    type=float,
    help='normal variance for offline sampling')
parser.add_argument(
    '--job_sample',
    default='uniform',
    help='sampling method for offline scheduling (uniform | normal)')
parser.add_argument(
    '--cache',
    type=int,
    default=0,
    help='number of local client models to keep')
parser.add_argument(
    '--cache_assist_wt',
    type=float,
    default=0.05,
    help='local model cache pre-softmax application factor')
parser.add_argument(
    '--test_steps',
    type=int,
    default=1000,
    help='steps between test model updates (default: 1000)')
parser.add_argument(
    '--workers',
    type=int,
    default=mpi.size - 2,
    metavar='W',
    help='how many training processes to use (default: 32)')
parser.add_argument(
    '--num-steps',
    type=int,
    default=20,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=10000,
    metavar='M',
    help='maximum length of an episode (default: 10000)')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')
parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')
parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')
parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')

parser.add_argument(
    '--clients',
    default=128,
    type=int,
    help='simulated federation size')

parser.add_argument(
    '--experiment',
    default=None,
    help='the parent experiment name (default: none)')

parser.add_argument(
    '--name',
    default='noname',
    help='the run name')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--method',
    default='merge_delta',
    help='aggregation strategy')

parser.add_argument(
    '--merge_wt',
    default='poisson_raw_scaled',
    help='merge weight strategy')
parser.add_argument(
    '--merge_max',
    default=1,
    help='max merge weight')
parser.add_argument(
    '--merge_min',
    default=0,
    help='min merge weight')

parser.add_argument(
    '--age_calc',
    default='len',
    help='method to track global age')

parser.add_argument(
    '--report',
    default='global_age',
    help='reporting X axis')

parser.add_argument(
    '--potential_mdp_cap',
    default=0.25,
    type=float,
    help='potential midpoint upper bound client wt')
parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    if args.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    # scheduler thread
    if mpi.rank == 0:
        sys.exit(scheduler(args, shared_model, env_conf))

    # testing thread
    if mpi.rank == 1:
        sys.exit(test(args, shared_model, env_conf))

    # spawn workers
    sys.exit(train(args, shared_model, env_conf))
