import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
from parameters import Parameters
from core import replay_memory
from core.mod_utils import is_lnorm_key
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class GeneticAgent:
    def __init__(self, args: Parameters):

        self.args = args

        self.actor = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)

        self.buffer = replay_memory.ReplayMemory(self.args.individual_bs, args.device)
        self.loss = nn.MSELoss()

    def update_params_crossover(self, batch, p1, p2, critic):
        state_batch, _, _, _, _ = batch

        p1_action = p1(state_batch)
        p2_action = p2(state_batch)
        p1_q = critic(state_batch, p1_action).flatten()
        p2_q = critic(state_batch, p2_action).flatten()

        eps = 0.0
        action_batch = torch.cat((p1_action[p1_q - p2_q > eps], p2_action[p2_q - p1_q >= eps])).detach()
        state_batch = torch.cat((state_batch[p1_q - p2_q > eps], state_batch[p2_q - p1_q >= eps]))
        actor_action = self.actor(state_batch)

        # Actor Update
        self.actor_optim.zero_grad()
        sq = (actor_action - action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()

    def update_params_mutation(self, actor_action, target_action_batch):
        # Actor Update
        self.actor_optim.zero_grad()
        sq = (actor_action - target_action_batch)**2
        policy_loss = torch.sum(sq) + torch.mean(actor_action**2)
        policy_mse = torch.mean(sq)
        policy_loss.backward()
        self.actor_optim.step()

        return policy_mse.item()


class Actor(nn.Module):

    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        l1 = self.args.ls1; l2 = self.args.ls2; l3 = l2

        # Construct Hidden Layer 1
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        # Out
        self.w_out = nn.Linear(l3, args.action_dim)

        self.to(self.args.device)

    def forward(self, input):

        # Hidden Layer 1
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = out.tanh()

        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = out.tanh()

        # Out
        out = (self.w_out(out)).tanh()
        return out

    def select_action(self, env, state, time_step, warm_up):
        if time_step < warm_up:
            mu = torch.tensor(env.action_space.sample()).to(self.args.device)
            return mu.cpu().data.numpy().flatten()
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.args.device)
            return self.forward(state).cpu().data.numpy().flatten()

    def get_novelty(self, batch):
        state_batch, action_batch, _, _, _ = batch
        novelty = torch.mean(torch.sum((action_batch - self.forward(state_batch))**2, dim=-1))
        return novelty.item()

    # function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.grad.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = torch.zeros(tot_size, dtype=torch.float32).to(self.args.device)
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            pvec[count:count + sz] = param.view(-1)
            count += sz
        return pvec.detach().clone()

    # function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            sz = param.numel()
            raw = pvec[count:count + sz]
            reshaped = raw.view(param.size())
            param.data.copy_(reshaped.data)
            count += sz

    # count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for name, param in self.named_parameters():
            if is_lnorm_key(name) or len(param.shape) != 2:
                continue
            count += param.numel()
        return count


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = self.args.ls1; l2 = self.args.ls2; l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_l1 = nn.Linear(args.state_dim + args.action_dim, l1)

        # Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)

        # Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        self.to(self.args.device)

    def forward(self, input, action):

        # Hidden Layer 1 (Input Interface)
        out = F.elu(self.w_l1(torch.cat((input, action),1)))

        # Hidden Layer 2
        out = self.w_l2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out


class DARC(object):
    def __init__(self, args):

        self.args = args
        self.buffer = replay_memory.ReplayMemory(args.individual_bs, args.device)

        self.critic1 = Critic(args)
        self.critic1_target = Critic(args)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=1e-3)

        self.critic2 = Critic(args)
        self.critic2_target = Critic(args)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=1e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.policy_noise = self.args.policy_noise
        self.noise_clip = self.args.noise_clip
        self.q_weight = self.args.q_weight
        self.regularization_weight = self.args.regularization_weight
        self.loss = nn.MSELoss()

        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

    def update_parameters(self, batch, actor1, actor2, actor1_target, actor2_target, actor1_optim, actor2_optim):
        self.update_one_q(batch, True, actor1, actor1_target, actor2_target, actor1_optim)
        self.update_one_q(batch, False, actor2, actor1_target, actor2_target, actor2_optim)
    
    def update_one_q(self, batch, update_a1, actor, actor1_target, actor2_target, actor_optim):
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = batch

        # Load everything to GPU if not already
        actor.to(self.args.device)
        actor1_target.to(self.args.device)
        actor2_target.to(self.args.device)
        self.critic1_target.to(self.args.device)
        self.critic2_target.to(self.args.device)
        self.critic1.to(self.args.device)
        self.critic2.to(self.args.device)

        state_batch = state_batch.to(self.args.device)
        next_state_batch = next_state_batch.to(self.args.device)
        action_batch = action_batch.to(self.args.device)
        reward_batch = reward_batch.to(self.args.device)
        if self.args.use_done_mask: done_batch = done_batch.to(self.args.device)

        with torch.no_grad():
            next_action1_batch = actor1_target(next_state_batch)
            next_action2_batch = actor2_target(next_state_batch)

            noise = torch.randn(
				(action_batch.shape[0], action_batch.shape[1]), 
				dtype=action_batch.dtype, layout=action_batch.layout, device=action_batch.device
			) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action1_batch = (next_action1_batch + noise).clamp(self.args.action_low, self.args.action_high)
            next_action2_batch = (next_action2_batch + noise).clamp(self.args.action_low, self.args.action_high)

            next_q1_a1 = self.critic1_target.forward(next_state_batch, next_action1_batch)
            next_q2_a1 = self.critic2_target.forward(next_state_batch, next_action1_batch)

            next_q1_a2 = self.critic1_target.forward(next_state_batch, next_action2_batch)
            next_q2_a2 = self.critic2_target.forward(next_state_batch, next_action2_batch)

            ## Min first, max afterward to avoid underestimation bias
            next_q1 = torch.min(next_q1_a1, next_q2_a1)
            next_q2 = torch.min(next_q1_a2, next_q2_a2)

            ## Soft q update
            next_q = self.q_weight * torch.min(next_q1, next_q2) + (1-self.q_weight) * torch.max(next_q1, next_q2)

            if self.args.use_done_mask: next_q = next_q * (1 - done_batch) #Done mask
            target_q = reward_batch + (self.gamma * next_q).detach()

        if update_a1:
            # Critic Update
            current_q1 = self.critic1(state_batch, action_batch)
            current_q2 = self.critic2(state_batch, action_batch)
			    
            critic1_loss = F.mse_loss(current_q1, target_q) + self.regularization_weight * F.mse_loss(current_q1, current_q2)

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            nn.utils.clip_grad_norm_(self.critic1.parameters(), 10)
            self.critic1_optim.step()

            actor1_loss = -self.critic1(state_batch, actor(state_batch)).mean()

            actor_optim.zero_grad()
            actor1_loss.backward()
            actor_optim.step()

            soft_update(actor1_target, actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
        else:
            # Critic Update
            current_q1 = self.critic1(state_batch, action_batch)
            current_q2 = self.critic2(state_batch, action_batch)
			    
            critic2_loss = F.mse_loss(current_q2, target_q) + self.regularization_weight * F.mse_loss(current_q2, current_q1)

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            nn.utils.clip_grad_norm_(self.critic2.parameters(), 10)
            self.critic2_optim.step()

            actor2_loss = -self.critic2(state_batch, actor(state_batch)).mean()

            actor_optim.zero_grad()
            actor2_loss.backward()
            actor_optim.step()

            soft_update(actor2_target, actor, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)


def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta