import hydra
import numpy as np
import omegaconf
import torch
import mbrl.env.mujoco_envs
import mbrl.env.termination_fns
import mbrl.env.reward_fns
import mbrl.env.pets_cartpole_morph
import mbrl.algorithms.pets_morph as pets_morph


@hydra.main(config_path="exp/conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env = mbrl.env.pets_cartpole_morph.CartPoleMorphEnv()
    term_fn = mbrl.env.termination_fns.no_termination
    reward_fn = mbrl.env.reward_fns.cartpole_pets
    cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.algorithm.agent.ol_optimizer_cfg.device = cfg.device
    cfg.algorithm.agent.il_optimizer_cfg.device = cfg.device

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    return pets_morph.train(env, term_fn, reward_fn, cfg)


if __name__ == "__main__":
    run()