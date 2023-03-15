import mpi

import numpy as np
import time
import sys

from environment import atari_env

from model import A3Clstm

def scheduler(args, shared_model, env_conf):
    sched_rng = np.random.default_rng(seed=args.seed*100 + 130)
    env = atari_env(args.env, env_conf, args)
    global_model = A3Clstm(env.observation_space.shape[0], env.action_space)
    global_age = 0
    global_parameters = global_model.state_dict().copy()
    client_parameters = [global_parameters.copy() for _ in range(mpi.size - 1)]

    # wait for messages
    while True:
        msg, payload = mpi.comm.recv()

        # get the next scheduling step
        if msg == 'schedule':
            source = payload

            step_range = args.max_offline_steps - args.min_offline_steps
            total_steps = sched_rng.integers(step_range) + args.min_offline_steps

            response = (global_age, total_steps, global_parameters.copy())
            mpi.comm.send(('next_schedule', response), dest=source)

        # send global model to requesting clients
        elif msg == 'get_global_model':
            source = payload
            parameters = global_parameters.copy()
            mpi.comm.send(('global_model', (parameters, global_age)), dest=source)

        # update local client model list
        elif msg == 'update_client_model':
            source, client_model = payload
            client_parameters[source] = client_model

        # retrieve a random past client model
        elif msg == 'get_random_client_model':
            source = payload

            available_params = [c for i, c in enumerate(client_parameters) if i != source]
            available_params = [p for p in available_params if p is not None]

            if len(available_params) > 0:
                response_params = np.random.choice(available_params)
            else:
                response_params = None

            mpi.comm.send(('random_client_model', response_params), dest=source)

        # update global parameters, with some midpoint weight
        elif msg == 'update_global_model':
            start, length, params, delta = payload

            if start + length == global_age:
                # the client is equal, we get the exact midpoint
                potential = 0
            elif start + length > global_age:
                #print(start, length, global_age)
                # the client is older, have 1 <=> client twice as old
                potential = -1 + (start + length) / max(1, global_age)
            else:
                # the client is younger, have -1 <=> client twice as young
                potential = 1 + -global_age / max(start + length, 1)

            # activate potential, scale between -1 (global only) and 1 (client only)
            potential = np.tanh(potential)

            #potential = 0.5
            #print('update potential', potential)

            # use as linear weight
            client_wt = (potential + 1) / 2

            # don't allow single clients to drastically change the model
            client_wt = min(args.potential_mdp_cap, client_wt)

            new_global_params = {}
            global_wt = 1 - client_wt

            #print('global model will update')
            kk = list(delta.keys())[0]
            #print(global_parameters[kk], delta[kk])

            # for discarding method, drop off less relevant weight updates
            # used only in discard method
            discard_wt = (start + length) / global_age if start + length < global_age else 1

            # discarding relative to the start age
            discard_rel_wt = length / (global_age - start) if start + length < global_age else 1

            for k in global_parameters.keys():
                if args.method == 'potential_delta_full':
                    new_global_params[k] = global_parameters[k] + delta[k]
                elif args.method == 'potential_delta_norm':
                    new_global_params[k] = global_parameters[k] + delta[k] / length
                elif args.method == 'potential_delta_age':
                    new_global_params[k] = global_parameters[k] + delta[k] * client_wt
                elif args.method == 'potential_midpoint':
                    cv = params[k] * client_wt
                    gv = global_parameters[k] * global_wt

                    cv = cv.cpu()
                    new_global_params[k] = cv + gv
                elif args.method == 'potential_discard':
                    new_global_params[k] = global_parameters[k] + delta[k] * discard_wt
                elif args.method == 'discard_rel':
                    new_global_params[k] = global_parameters[k] + delta[k] * discard_rel_wt
                else:
                    raise NotImplementedError(args.method)

                #print('from', global_parameters[k], 'to', new_global_params[k])

            global_parameters = new_global_params
            #print('global model updates')
            #print(global_parameters[kk])

            # be optimistic about the true global age (TODO this needs verifying)
            #global_age = max(global_age, start + length)
            global_age += length

        # retrieve a singular client model
        elif msg == 'get_client_model':
            source = payload
            mpi.send(('client_model', client_parameters[source].copy()), dest=source)

        # retrieve some sample from the client models
        elif msg == 'sample_client_models':
            source, count = payload
            models = np.random.choice(client_parameters, size=min(count, len(client_parameters)))

            mpi.send(('client_models', models), dest=source)

        # terminate the scheduler
        elif msg == 'stop':
            sys.exit(0)

        else:
            raise NotImplementedError(msg)
