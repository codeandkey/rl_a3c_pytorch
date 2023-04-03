import mpi

import numpy as np
import time
import sys

from environment import atari_env
from collections import deque

import scipy

from model import A3Clstm
from player_util import Agent
import math

import torch

def scheduler(args, shared_model, env_conf):
    sched_rng = np.random.default_rng(seed=args.seed*100 + 130)

    # simulated clients
    client_agents = [None for _ in range(args.clients)]
    client_optimizer_params = [None for _ in range(args.clients)]

    for player in client_agents:
        # initialize local agents

        player.model = A3Clstm(player.env.observation_space.shape[0],
                               player.env.action_space)

        player.state = player.env.reset()
        player.state = torch.from_numpy(player.state).float()
        player.model.train()
        player.eps_len += 2
    
    # global model
    ref_environment = atari_env(args.env, env_conf, args)
    state_space = ref_environments.observation_space.shape[0]
    action_space = ref_environments.action_space
    global_model = A3Clstm(state_space, action_space)
    global_parameters = global_model.state_dict()

    # status log timer
    last_log_time = time.time()

    # Initialize job generator RNGs for each simulated client
    job_rngs = [np.random.default_rng(
        seed=args.seed*args.clients + i) for i in range(args.clients)]

    # Initialize job status for each simulated client
    def next_job(client_rng, current_time):
        if args.job_sample == 'uniform':
            step_range = args.max_steps - args.min_steps
            delay_range = args.max_delay - args.min_delay
            start_delay = client_rng.integers(delay_range) + args.min_delay
            steps = client_rng.integers(step_range) + args.min_steps
        elif args.job_sample == 'normal':
            start_delay = int(args.delay_mean +
                              args.delay_var * client_rng.normal())
            start_delay = max(args.min_delay, start_delay)
            start_delay = min(args.max_delay, start_delay)

            steps = int(args.steps_mean + args.steps_var * client_rng.normal())
            steps = max(args.min_steps, steps)
            steps = min(args.max_steps, steps)
        else:
            raise NotImplementedError(args.job_sample)

        return {
            'start': current_time + start_delay,
            'end': current_time + start_delay + steps,
            'window': steps,
            'status': 'start_pending',
        }

    jobs = [next_job(job_rngs[i], 0) for i in range(args.clients)]
    current_time = 0
    total_updates = 0

    def update_global_model(client, delta_params):
        """Update the global model with the client's parameter update."""

        # increment the total number of updates
        nonlocal total_updates
        total_updates += 1

        # compute the current average window
        avg_client_window = sum(
            list(map(lambda x: x['window'], jobs))) / max(1, len(jobs))

        if avg_client_window == 0:
            merge_wt = 1
        else:
            window = jobs[client]['window']

            # scale delta update by scaled poisson distribution
            pmf_max = scipy.stats.poisson.pmf(
                math.floor(avg_client_window), avg_client_window)
            merge_wt = scipy.stats.poisson.pmf(
                window, avg_client_window) / pmf_max

            if args.merge_wt == 'poisson_raw_scaled':
                pass  # more compat with old experiments
            elif args.merge_wt == 'drop':
                merge_wt = 1 if window < avg_client_window else 0
            elif args.merge_wt == 'full':
                merge_wt = 1
            else:
                raise NotImplementedError(args.merge_wt)

            # merge the client's update into the global model
            for key in global_parameters.keys():
                global_parameters[key] = global_parameters[key] + \
                    delta_params[key] * merge_wt

    test_thread_waiting = False # true if test thread wants the params

    # in an infinite loop, we simulate each timestep at the server side.
    # before advancing to the next timestep, we must ensure all jobs overlapping
    # the current time have been completed.
    while current_time < args.max_time:
        # first, we iterate through jobs and collect those that are ready to start
        # or are required to finish before we can advance to the next timestep

        pending_result = []  # we need to wait for these jobs to finish before advancing
        pending_start = []  # we need to send these jobs to clients before advancing

        for client in range(args.clients):
            job = jobs[client]

            if job['start'] != current_time and job['end'] != current_time:
                # this job is not relevant to the current timestep
                continue

            if job['status'] == 'start_pending':
                # this job is ready to start
                pending_start.append(client)

            elif job['status'] == 'running':
                # this job is not complete yet
                pending_result.append(client)

            elif job['status'] == 'completed':
                # this job is completed, and destined to be merged
                update_global_model(client, job['delta_params'])

                # advance the job, don't allow immediate start
                jobs[client] = next_job(job_rngs[client], current_time + 1)

            else:
                raise NotImplementedError(job['status'])

        # write status log
        if time.time() - last_log_time > args.log_interval or True:
            print('t: {}, {} join, {} leave'.format(current_time,
                  len(pending_start), len(pending_result)))
            last_log_time = time.time()

        # if there are no pending jobs, we can advance to the next timestep
        if len(pending_result) == 0 and len(pending_start) == 0:
            current_time += 1
            continue

        if test_thread_waiting:
            # if the test thread is waiting for the global model, we can send it
            # now
            test_thread_waiting = False
            mpi.comm.send(('global_model', (current_time, global_parameters, total_updates)), dest=1)

        # we continue processing messages until both pending lists are empty
        while len(pending_result) > 0 or len(pending_start) > 0:
            # process the next message received
            msg, payload = mpi.comm.recv()

            if msg == 'get_global_model':
                # if all jobs pending result are done, we can send the global 
                # model, otherwise we send a dummy model and reply to the test
                # thread on the next timestep

                source = payload

                if len(pending_result) == 0:
                    mpi.comm.send(('global_model', (current_time, global_parameters, total_updates)), dest=source)
                else:
                    test_thread_waiting = True

            elif msg == 'sync':
                # a client will interact with the server
                # we receive the client's parameter update (if any), and respond
                # with the current global parameters and a job to run

                source, update_params = payload

                if update_params:
                    # receive update parameters from client
                    client = update_params['client']
                    delta_params = update_params['delta_params']
                    agent = update_params['agent']
                    optimizer_params = update_params['optimizer_params']

                    # update local client states
                    client_agents[client] = agent
                    client_optimizer_params[client] = optimizer_params

                    # get next job, starting next timestep at earliest
                    jobs[client] = next_job(
                        job_rngs[client], current_time + 1)

                    if client not in pending_result:
                        # we've received an early result, we write the job as completed
                        # and wait for the future merge

                        # sanity check, this result should be for a future job
                        assert current_time < jobs[client]['end']

                        # write the job as completed
                        jobs[client]['status'] = 'completed'

                        # save the params for a later timestep
                        jobs[client]['delta_params'] = delta_params

                    else:
                        pending_result.remove(client)

                        # update the global model with the client's parameter update
                        update_global_model(
                            client, delta_params)

                # build the next job for this client
                new_job_params = None

                if len(pending_start) > 0:
                    new_job_client = pending_start.pop()
                    new_job_params = {
                        'client': new_job_client,
                        'steps': jobs[new_job_client]['window'],
                        'model': global_parameters.copy(),
                        'agent': client_agents[new_job_client],
                        'params': global_parameters.copy(),
                        'optimizer_params': client_optimizer_params[new_job_client],
                    }
                    jobs[new_job_client]['status'] = 'running'

                mpi.comm.send(('go', new_job_params), dest=source)

            # terminate the scheduler
            elif msg == 'stop':
                sys.exit(0)

            else:
                raise NotImplementedError(msg)
