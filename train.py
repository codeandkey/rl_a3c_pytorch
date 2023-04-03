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

def train(args, shared_model, env_conf):
    ptitle('Training Agent: {}'.format(mpi.rank))
    gpu_id = args.gpu_ids[mpi.rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + mpi.rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + mpi.rank)
    env = atari_env(args.env, env_conf, args)
    env.seed(args.seed + mpi.rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2

    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(player.model.parameters(), lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            player.model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    updated_model = False
    last_global = {}
    model_cache = [A3Clstm(player.env.observation_space.shape[0], player.env.action_space) for _ in range(args.cache)]
    
    start = 0
    length = 0

    while True:
        if updated_model:
            # send next model update
            params = player.model.state_dict().copy()
            delta = { k: params[k].cpu() - last_global[k].cpu() for k in params.keys() }

            mpi.comm.send(('update_global_model', (start, length, params, delta, mpi.rank)), dest=0)

        # reload cache
        if args.cache > 0:
            for i in range(args.cache):
                mpi.comm.send(('get_random_client_model', mpi.rank), dest=0)

                msg, payload = mpi.comm.recv(source=0)

                if msg != 'random_client_model':
                    raise RuntimeError('unexpected ' + msg)

                model_cache[i].load_state_dict(payload)

        # get schedule step
        mpi.comm.send(('schedule', mpi.rank), dest=0)

        msg, payload = mpi.comm.recv(source=0)

        if msg != 'next_schedule':
            raise RuntimeError('unexpected ' + msg)

        start, length, global_parameters = payload
        total_steps = 0
        updated_model = True # for the next iteration

        # load new local actor parameters
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(global_parameters)
        else:
            player.model.load_state_dict(global_parameters)

        last_global = global_parameters.copy()

        # ensure new parameters are optimized
        optimizer.param_groups[0]['params'] = player.model.parameters()

        #print(mpi.rank, 'training for', length)

        while total_steps < length:

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
                player.action_train(args, assist_models=model_cache)
                total_steps += 1
                if player.done or total_steps >= length:
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
