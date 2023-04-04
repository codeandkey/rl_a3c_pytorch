from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging

import sys
import os

import mpi

def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    # initialize output files

    basepath = 'results/'

    if args.experiment is not None:
        basepath += args.experiment + '/'

    os.makedirs(basepath, exist_ok=True)
    outpath = basepath + args.name

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.process_time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    total_steps = 0
    report_rewards = []
    report_times = []
    report_ages = []
    while True:
        # get global model from scheduler
        mpi.comm.send(('get_global_model', mpi.rank), dest=0)

        msg, payload = mpi.comm.recv(source=0)

        if msg == 'global_model':
            current_time, global_parameters, total_updates = payload

            # check the parameters actually changed

            pmsd = player.model.cpu().state_dict()
            k1 = list(pmsd.keys())[0]
            print('test model update diff', global_parameters[k1] - pmsd[k1])

            player.model.load_state_dict(global_parameters.copy())

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.cuda()
            #print('test model updates')
        elif msg == 'stop':
            sys.exit(0)
        else:
            raise NotImplementedError(msg)

        #print('testing for',args.test_steps)
        #print('first param', player.model.state_dict()['actor_linear.weight'])

        for i in range(args.test_steps):
            player.action_test()
            reward_sum += player.reward

            if player.done and not player.info:
                state = player.env.reset()
                player.eps_len += 2
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
            elif player.info:
                flag = True
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                #reward_mean = sum(test_results[-args.window:]) / len(test_results[-args.window:])

                report_rewards.append(reward_sum)
                report_times.append(current_time)
                report_ages.append(total_updates)

                with open(outpath, 'w') as f:
                    f.write(str({'age': report_ages, 'reward': report_rewards, 'time': report_times}))

                log['{}_log'.format(args.env)].info(
                        "Time {0}, timestep {5}, age {4}, ep reward {1}, ep length {2}, reward mean {3:.4f}".
                    format(
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.process_time() - start_time)),
                        reward_sum, player.eps_len, reward_mean, total_updates, current_time))

                if args.save_max and reward_sum >= max_score:
                    max_score = reward_sum
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(state_to_save, '{0}{1}.dat'.format(
                                args.save_model_dir, args.env))
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))

                reward_sum = 0
                player.eps_len = 0
                state = player.env.reset()
                player.eps_len += 2

                # this is cursed, let's just blow up result data instead
                #time.sleep(10)
                #start_time -= 10 # as time.process_time() excludes waiting time

                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
