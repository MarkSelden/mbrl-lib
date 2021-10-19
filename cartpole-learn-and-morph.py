import copy
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import mbrl.env.termination_fns as termination_fns
import mbrl.env.reward_fns as reward_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import gym
import mbrl.env.cartpole_continuous_morph
import mbrl.util.logger
import mbrl
from datetime import datetime
#os.environ['HYDRA_FULL_ERROR']='1'

@hydra.main(config_path="conf/morph_exps", config_name="cfg_dict")
def main(cfg: DictConfig):
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
    reward_fn = reward_fns.PETS_cartpole_morph
    #reward_fn = reward_fns.cartpole_morph
    trial_length = 200
    num_trials = 100
    cfg.dynamics_model.model.device = device
    cfg.overrides.trial_length = trial_length
    cfg.overrides.num_steps = num_trials * trial_length
    cfg.agent.ol_optimizer_cfg.device = device
    cfg.agent.il_optimizer_cfg.device = device



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




    agent = planning.create_morph_trajectory_optim_agent_for_model(
        model_env,
        cfg.agent,
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
    now = datetime.now()
    date = now.strftime('20%y-%m-%d')
    time = now.strftime("%H-%M-%S")
    cwd = os.getcwd()
    logger_name = f'morph-results-cartpole-toy'
    log_format =  [
        ("env_step", "S", "int"),
        ("episode_reward", "R", "float"),
        ("cart_mass", "C", "float"),
        ("pole_mass", "P", "float"),
        ("pole_length", "L", "float")

    ]
    logger = mbrl.util.Logger(cwd)
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
                if trial % 5 == 0:
                    #save the last model of the generation
                    dynamics_model.save(cwd)
                dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

                dataset_train, dataset_val = replay_buffer.get_iterators(
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    train_ensemble=True,
                    ensemble_size= cfg.dynamics_model.model.ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                model_trainer.train(
                    dataset_train, dataset_val=dataset_val, num_epochs=cfg.dynamics_model.num_epochs, patience=cfg.dynamics_model.patience, callback=train_callback)

                #check if its time to create a new morphology, mod 5?
                if trial % 5 ==0:
                    agent.morph_again()



            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = common_util.morph_step_env_and_add_to_buffer(env, obs, agent, {}, replay_buffer)

            #update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            if steps_trial == trial_length:
                break
        cart_m, pole_m, pole_l = env.get_exp_params()
        logger.log_data(logger_name, {"env_step": steps_trial, "episode_reward": total_reward, "cart_mass": cart_m, "pole_mass": pole_m, "pole_length": pole_l},)



main()