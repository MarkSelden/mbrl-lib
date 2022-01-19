import os

import numpy as np
import torch
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
import xmltodict


class CartPoleMorphEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    TARGET = 0.6

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        initial_pole_length = [np.random.rand() * 5]
        self.asset_file = "%s/assets/cartpole_morph.xml" % self.dir_path
        self.set_exp_params(initial_pole_length)
        self.exp_param_space = spaces.Box(np.array([0.001]), np.array([20]))
        mujoco_env.MujocoEnv.__init__(self, self.asset_file, 2)





    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartPoleMorphEnv.TARGET
        reward = np.exp(
            -np.sum(
                np.square(
                    self._get_ee_pos(ob) - np.array([0.0, CartPoleMorphEnv.TARGET])
                )
            )
            / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.append(np.array([self.pole_length]), np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel())

    def _get_ee_pos(self, x):
        x0, theta = x[0], x[1]
        return np.array(
            [
                x0 - self.pole_length * np.sin(theta),
                -self.pole_length * np.cos(theta),
            ]
        )

    def set_exp_params(self, params: np.ndarray):
        assert(len(params) == 1)

        new_length = params[0]
        #need to re adjust the pole length.

        with open(self.asset_file, 'r') as fd:
            xml_string = fd.read()
        xml_dict = xmltodict.parse(xml_string)
        # This points to the pole
        xml_dict['mujoco']['worldbody']['body']['body']['geom']['@fromto'] =  "0 0 0 0.001 0 -%f" % new_length

        xml_string = xmltodict.unparse(xml_dict, pretty=True)

        with open(self.asset_file, 'w') as fd:
            fd.write(xml_string)

        self.pole_length = new_length

        #I believe this should remorph but not 100%
        mujoco_env.MujocoEnv.__init__(self, self.asset_file, 2)

    @staticmethod
    def preprocess_fn(state):
        if isinstance(state, np.ndarray):
            return np.concatenate(
                [
                    np.sin(state[..., 1:2]),
                    np.cos(state[..., 1:2]),
                    state[..., :1],
                    state[..., 2:],
                ],
                axis=-1,
            )
        if isinstance(state, torch.Tensor):
            return torch.cat(
                [
                    torch.sin(state[..., 1:2]),
                    torch.cos(state[..., 1:2]),
                    state[..., :1],
                    state[..., 2:],
                ],
                dim=-1,
            )
        raise ValueError("Invalid state type (must be np.ndarray or torch.Tensor).")

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

