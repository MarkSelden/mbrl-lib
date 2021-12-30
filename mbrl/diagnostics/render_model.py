import argparse
import pathlib

import mbrl
import mbrl.models
import mbrl.planning
import mbrl.util.common
import mbrl.util.mujoco
import matplotlib.pyplot as plt
import numpy as np
import torch

class Agent_Renderer():

    def __init__(self, exp_dir, num_steps):
        self.max_steps = num_steps
        exp_path = pathlib.Path(exp_dir)
        self.cfg = mbrl.util.common.load_hydra_cfg(exp_path)
        self.env, self.term_fn, self.reward_fn = mbrl.util.mujoco.make_env(self.cfg)

        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=exp_path,
        )

        self.model_env = mbrl.models.ModelEnv(
            self.env,
            self.dynamics_model,
            self.term_fn,
            self.reward_fn,
            generator=torch.Generator(self.dynamics_model.device),
        )

        agent_cfg = self.cfg
        if (
                agent_cfg.algorithm.agent._target_
                == "mbrl.planning.TrajectoryOptimizerAgent"
        ):
            self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
                self.model_env,
                agent_cfg.algorithm.agent,
                num_particles=agent_cfg.algorithm.num_particles,
            )
        else:
            agent_cfg = mbrl.util.common.load_hydra_cfg(agent_dir)
            if (
                agent_cfg.algorithm.agent._target_
                == "mbrl.planning.TrajectoryOptimizerAgent"
            ):
                agent_cfg.algorithm.agent.planning_horizon = lookahead
                self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
                    self.model_env,
                    agent_cfg.algorithm.agent,
                    num_particles=agent_cfg.algorithm.num_particles,
                )


        # Set up recording
        #self.vis_path = self.exp_path / "render"


    def run_exp(self):

        steps = 0

        obs = self.env.reset()
        self.agent.reset()
        done = False
        total_reward = 0.0

        while not done and steps < self.max_steps:

            # --- Doing env step using the agent and adding to model dataset ---
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1
            self.env.render(mode="rgb_array")



        return np.float32(total_reward)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run. Should contain both the model and the configs.",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="The number of steps to render.",
    )

    args = parser.parse_args()
    experiment = Agent_Renderer(args.experiments_dir, args.num_steps)
    reward = experiment.run_exp()
    print(f"This experiment generated a cumulative reward of {reward}")

