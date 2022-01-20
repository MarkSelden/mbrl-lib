# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np
from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)

#remove the static obs from the termination argument
def PETS_cartpole_morph(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    # calculate the distance between the point we set and the
    dev = act.device
    if dev != 'cpu':
        next_obs = next_obs.cpu()
        act = act.cpu()
    length = np.array(next_obs[:, 2])
    pos = np.array(next_obs[:,3])
    vel = np.array(next_obs[:,4])
    ang = np.array(next_obs[:,5])
    ang_vel = np.array(next_obs[:,6])
    ang = ang * -1 + np.pi/2
    target = (0,1)
    hyp = 2 * length
    pole_edge  = np.array([hyp * np.cos(ang) + pos, hyp * np.sin(ang)])
    target_distance = np.sqrt((pole_edge[0] - target[0])**2 + (pole_edge[1] - target[1])**2)
    reward = target_distance * -1 + (-0.01 * np.array(act).flatten()**2)
    return torch.tensor(np.reshape(reward, newshape=(len(reward),1)), device=dev)

def cartpole_morph(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole_morph(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6 ** 2))
    act_cost = -0.01 * torch.sum(act ** 2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)

#this is the mujoco one.
def cartpole_pets_morph(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    pole_len = next_obs[:, :1]
    x0 = next_obs[:, 1:2]
    theta = next_obs[:, 2:3]
    ee_pos = torch.cat([x0 - pole_len * theta.sin(), -pole_len * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6 ** 2))
    act_cost = -0.01 * torch.sum(act ** 2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)
