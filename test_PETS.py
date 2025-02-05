
import hydra
import numpy as np
import omegaconf
import torch
import mbrl.env.cartpole_continuous
import mbrl.env.termination_fns
import mbrl.env.reward_fns
import mbrl.algorithms.pets as pets

@hydra.main(config_path="mbrl/examples/conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env = mbrl.env.mujoco_envs.CartPoleEnv()
    term_fn = mbrl.env.termination_fns.no_termination
    reward_fn = None
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    return pets.train(env, term_fn, reward_fn, cfg)


if __name__ == "__main__":
    run()