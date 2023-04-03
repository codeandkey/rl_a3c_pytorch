from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable

import mpi
import time
import copy

def train(args, env_conf):
    ptitle('train {}'.format(mpi.rank))
    gpu_id = args.gpu_ids[mpi.rank % len(args.gpu_ids)]

    model = None # you better not use this model

    optimizer = None
    update_params = None

    while True:
        # sync with the server
        mpi.comm.send(('sync', (mpi.rank, update_params)), dest=0)

        # receive next work
        msg, payload = mpi.comm.recv(source=0)

        if msg == 'stop':
            break

        if msg != 'go':
            raise NotImplementedError('unexpected ' + msg)

        if payload is None:
            # no work to do, wait a bit
            time.sleep(0.1)
            continue

        client = payload['client']
        steps = payload['steps']
        params = payload['params'].copy()
        player = payload['agent']
        optimizer_params = payload['optimizer_params']

        if not player:
            # the environment is not initialized yet, we do it here
            # to enable parallelism
            env = atari_env(args.env, env_conf, args)
            env.seed(args.seed * 340 + client)
            player = Agent(None, env, args, None)
            player.gpu_id = gpu_id
            player.state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(player.state).float()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        if not model:
            model = A3Clstm(player.env.observation_space.shape[0],
                            player.env.action_space)

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    model = model.cuda()

        # reload player model
        player.model = model

        # update local model
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(params)
        else:
            player.model.load_state_dict(params)

        player.model.train()

        total_steps = 0
        last_global_params = params.copy()

        # ensure new parameters are optimized
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(player.model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                player.model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(player.model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError(args.optimizer)

        # update optimizer state
        if optimizer_params:
            optimizer.load_state_dict(optimizer_params.copy())

        # update optimizer parameter group (again)
        #optimizer.param_groups[0]['params'] = player.model.parameters()

        #print(mpi.rank, 'training for', length)

        while total_steps < steps:

            if player.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.cx = Variable(torch.zeros(1, 512).cuda())
                        player.hx = Variable(torch.zeros(1, 512).cuda())
                else:
                    player.cx = Variable(torch.zeros(1, 512))
                    player.hx = Variable(torch.zeros(1, 512))
            else:
                player.cx = Variable(player.cx.data)
                player.hx = Variable(player.hx.data)

            for step in range(args.num_steps):
                player.action_train(args)
                total_steps += 1
                if player.done or total_steps >= steps:
                    break

            if player.done:
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()

            R = torch.zeros(1, 1)
            if not player.done:
                value, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                            (player.hx, player.cx)))
                R = value.data

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = R.cuda()

            player.values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    gae = gae.cuda()
            R = Variable(R)
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + args.gamma * \
                    player.values[i + 1].data - player.values[i].data

                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    player.log_probs[i] * \
                    Variable(gae) - 0.01 * player.entropies[i]

            player.model.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            #ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()
            player.clear_actions()

        # set update params for the next sync
        params = player.model.state_dict().copy()
        delta_params = { k: params[k].cpu() - last_global_params[k].cpu() for k in params.keys() }

        # we can bundle the env with the player, but we remove
        # the model from the player to save bandwidth

        player.model = None

        update_params = {
            'client': client,
            'delta_params': delta_params,
            'agent': player,
            'optimizer_params': optimizer.state_dict()
        }

