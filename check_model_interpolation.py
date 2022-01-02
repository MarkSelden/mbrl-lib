import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
import mbrl.env.termination_fns as termination_fns
import mbrl.env.reward_fns as reward_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import pybullet_envs
import gym
import mbrl.env.cartpole_continuous_morph
import pandas as pd

mpl.rcParams.update({"font.size": 16})

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

seed = 16
env = mbrl.env.cartpole_continuous_morph.CartPoleMorphEnv()
env.seed(seed)
rng = np.random.default_rng(seed=0)
#I believe the generator is used for random sampling
generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
exp_param_space = env.exp_param_space

# This functions allows the model to evaluate the true rewards given an observation
# This function allows the model to know if an observation should make the episode end
term_fn = termination_fns.cartpole_morph
reward_fn = reward_fns.cartpole_morph

trial_length = 200
num_trials = 10
ensemble_size = 5

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "model": {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "use_silu": True,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model"
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": False,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials * trial_length,
        "model_batch_size": 1,
        "validation_ratio": 0.05
    }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

# Create a dynamics model for this environment
dynamics_model = util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)

replay_buffer = util.common.create_replay_buffer(
    cfg, obs_shape, act_shape, rng=rng
)

util.common.rollout_agent_trajectories(
    env,
    100,
    planning.RandomMorphingAgent(env),
    {},
    replay_buffer=replay_buffer,
)


# Create a trainer for the model
model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

dataset_train, dataset_val = replay_buffer.get_iterators(
    batch_size=cfg.overrides.model_batch_size,
    val_ratio=cfg.overrides.validation_ratio,
    train_ensemble=True,
    ensemble_size=ensemble_size,
    shuffle_each_epoch=True,
    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
)

model_trainer.train(
    dataset_train, dataset_val=dataset_val, num_epochs=50, patience=50, callback=None)


for i in np.linspace(start = exp_param_space.low, stop  = exp_param_space.high):

    