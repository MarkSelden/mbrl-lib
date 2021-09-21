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
    target = (0,5)
    hyp = 2 * length
    pole_edge  = np.array([hyp * np.cos(ang) + pos, hyp * np.sin(ang)])
    target_distance = np.sqrt((pole_edge[0] - target[0])**2 + (pole_edge[1] - target[1])**2)
    reward = target_distance * -1 + (-0.01 * np.array(act).flatten()**2)
    return torch.tensor(np.reshape(reward, newshape=(300,1)), device=dev)

def cartpole_morph(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole_morph(act, next_obs)).float().view(-1, 1)


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
