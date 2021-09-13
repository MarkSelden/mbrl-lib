import os
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
import gym
import mbrl.env.cartpole_continuous_morph
import mbrl.util.logger

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
num_trials = 100
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
    1,
    planning.RandomMorphingAgent(env),
    {},
    replay_buffer=replay_buffer,
)

# Create a gym-like environment to encapsulate the model
model_env = models.ModelEnv(
        env, dynamics_model, term_fn, reward_fn, generator=generator
    )


agent_cfg = omegaconf.OmegaConf.create({
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.OLOptimizerAgent",
    "planning_horizon": 10,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    "environment_params_lb": "???",
    "environment_params_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "ol_optimizer_cfg": {
        "_target_": "mbrl.planning.MorphCEMOptimizer",
        "num_iterations": 10,
        "elite_ratio": 0.1,
        "population_size": 20,
        "alpha": 0.1,
        "device": device,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True
    },
    "il_optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 10,
        "elite_ratio": 0.1,
        "population_size": 20,
        "alpha": 0.1,
        "device": device,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True
    }
})

agent = planning.create_morph_trajectory_optim_agent_for_model(
    model_env,
    agent_cfg,
    num_particles=15
)
train_losses = []
val_scores = []

def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
    train_losses.append(tr_loss)
    val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

def update_axes(_axs, _frame, _text, _trial, _steps_trial, _all_rewards, force_update=False):
    if not force_update and (_steps_trial % 10 != 0):
        return
    _axs[0].imshow(_frame)
    _axs[0].set_xticks([])
    _axs[0].set_yticks([])
    _axs[1].clear()
    _axs[1].set_xlim([0, num_trials + .1])
    _axs[1].set_ylim([0, 200])
    _axs[1].set_xlabel("Trial")
    _axs[1].set_ylabel("Trial reward")
    _axs[1].plot(_all_rewards, 'bs-')
    _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")



# Create a trainer for the model
model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

# Create visualization objects
#fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
#ax_text = axs[0].text(300, 50, "")

#Set up logger
logger_name = 'morph-results-0 Cartpole-toy'
log_format =  [
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
]
work_dir = os.getcwd()
logger = mbrl.util.Logger(work_dir)
logger.register_group(logger_name, log_format, color="green")

# Main PETS loop
all_rewards = [0]
for trial in range(num_trials):
    obs = env.reset()
    agent.reset()

    done = False
    total_reward = 0.0
    steps_trial = 0
    #update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)
    while not done:
        # --------------- Model Training -----------------
        if steps_trial == 0:
            #Add the reset for the planning of agent bodies in here
#            print('training model')
            dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

            dataset_train, dataset_val = replay_buffer.get_iterators(
                batch_size=cfg.overrides.model_batch_size,
                val_ratio=cfg.overrides.validation_ratio,
                train_ensemble=True,
                ensemble_size=ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
            )

            #TODO:: Retraining the model fully every time???

            model_trainer.train(
                dataset_train, dataset_val=dataset_val, num_epochs=20, patience=5, callback=train_callback)

            agent.reset_action_count()

        # --- Doing env step using the agent and adding to model dataset ---
        next_obs, reward, done, _ = common_util.morph_step_env_and_add_to_buffer(env, obs, agent, {}, replay_buffer)

        #update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

        obs = next_obs
        total_reward += reward
        steps_trial += 1
        if steps_trial == trial_length:
            break
    logger.log_data(logger_name, {"env_step": steps_trial, "episode_reward": total_reward},)


#update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards, force_update=True)
