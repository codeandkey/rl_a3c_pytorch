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


def scheduler(args, shared_model, env_conf):
    sched_rng = np.random.default_rng(seed=args.seed*100 + 130)
    global_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    global_parameters = global_model.state_dict().copy()

    state_space = client_environments[0].observation_space.shape[0]
    action_space = client_environments[0].action_space
    global_model = A3Clstm(state_space, action_space)
    global_model = A3Clstm

    client_environments = [atari_env(args.env, env_conf, args)
                           for _ in range(args.clients)]
    client_agents = [Agent(None, client_environments[i], args, None) for i in range(args.clients)]

    # Initialize job generator RNGs for each simulated client
    job_rngs = [np.random.default_rng(
        seed=args.seed*args.clients + i) for i in range(args.clients)]

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

    # Initialize job status for each simulated client
    jobs = [next_job(job_rngs[i], 0) for i in range(args.clients)]

    client_windows = np.zeros(mpi.size - 2)
    current_time = 0

    def update_global_model(client, window, delta_params):
        """Update the global model with the client's parameter update."""

        # compute the current average window
        avg_client_window = sum(
            list(map(lambda x: x['window'], jobs))) / max(1, len(jobs))

        if avg_client_window == 0:
            merge_wt = 1
        else:
            current_window = jobs[client]['end'] - jobs[client]['start']

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

    # status log timer
    last_log_time = time.time()

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

        # write status log
        if time.time() - last_log_time > args.log_interval:
           print(time.time() + ' t: {}, {} join, {} leave'.format(current_time,
                  len(pending_start), len(pending_result)))

        # if there are no pending jobs, we can advance to the next timestep
        if len(pending_result) == 0 and len(pending_start) == 0:
            current_time += 1
            continue

        # we continue processing messages until both pending lists are empty
        while len(pending_result) > 0 or len(pending_start) > 0:
            # process the next message received
            msg, payload = mpi.comm.recv()

            if msg == 'sync':
                # a client will interact with the server
                # we receive the client's parameter update (if any), and respond
                # with the current global parameters and a job to run

                source, update_params = payload

                if update_params is not None:
                    # receive update parameters from client
                    client, delta_params, agent, environment = update_params

                    # update the global model with the client's parameter update
                    update_global_model(
                        client, client_windows[client], delta_params)

                    # update local client states
                    client_agents[client] = agent
                    client_environments[client] = environment
                    client_windows[client] = jobs[client]['end'] - \
                        job[client]['start']

                # build the next job for this client
                new_job_params = None

                if len(pending_start) > 0:
                    new_job_client = pending_start.pop()
                    new_job_params = {
                        'client': new_job_client,
                        'model': global_parameters.copy(),
                        'agent': client_agents[new_job_client],
                        'environment': client_environments[new_job_client]
                    }

                mpi.comm.send(('go', new_job_params), dest=source)

            # terminate the scheduler
            elif msg == 'stop':
                sys.exit(0)

            else:
                raise NotImplementedError(msg)
