import time
from typing import Callable, List, Optional, Sequence, cast

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from .core import Agent, complete_agent_cfg
from .trajectory_opt import Optimizer, TrajectoryOptimizer

class RandomMorphingAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.counter = 0

    def act(self, *_args,  **_kwargs):
        if self.counter % 10 == 0:
            self.env.set_exp_params(self.env.exp_param_space.sample())

        return self.env.action_space.sample()

class MarksOptimizer(TrajectoryOptimizer):
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using th3 initial
            solution. Defaults to ``True``.
    """

    def __init__(
            self,
            optimizer_cfg: omegaconf.DictConfig,
            action_lb: np.ndarray,
            action_ub: np.ndarray,
            environmental_params_lb: np.ndarray,
            environmental_params_ub: np.ndarray,
            planning_horizon: int,
            replan_freq: int = 1,
            keep_last_solution: bool = True,
    ):
        lower_bounds =  np.tile(action_lb, (planning_horizon, 1)).tolist()
        upper_bounds =  np.tile(action_ub, (planning_horizon, 1)).tolist()
        optimizer_cfg.lower_bound = lower_bounds
        optimizer_cfg.upper_bound = upper_bounds
        optimizer_cfg.param_lb = environmental_params_lb.tolist()
        optimizer_cfg.param_ub = environmental_params_ub.tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        self.initial_solution = (
            ((torch.tensor(lower_bounds) + torch.tensor(upper_bounds)) / 2)
                .float()
                .to(optimizer_cfg.device)
        )
        self.initial_params = (
            ((torch.tensor(environmental_params_lb) + torch.tensor(environmental_params_lb)) / 2)
                .float()
                .to(optimizer_cfg.device)
        )
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.horizon = planning_horizon

    def optimize(
            self,
            trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
            callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (tuple of np.ndarray and float): the best action sequence.
        """

        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.previous_solution,
            p0=self.initial_params,
            callback=callback,
        )

        return (best_solution[0].cpu().numpy(), best_solution[1].cpu().numpy()) #hardcoded?

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()


#######################################


class MarksOptimizerAgent(Agent):
    def __init__(
            self,
            optimizer_cfg: omegaconf.DictConfig,
            optimizer_cfg2: omegaconf.DictConfig,
            action_lb: Sequence[float],
            action_ub: Sequence[float],
            environment_params_lb: Sequence[float],
            environment_params_ub: Sequence[float],
            initial_planning_horizon= 1,
            planning_horizon: int = 1,
            replan_freq: int = 1,
            verbose: bool = False,
    ):
        self.exp_param_size = len(environment_params_lb)
        self.first_optimizer = MarksOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            np.array(environment_params_lb),
            np.array(environment_params_ub),
            planning_horizon=initial_planning_horizon,
            replan_freq=replan_freq,
        )


        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg2,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }


        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose
        self.action_count = 0

    def set_trajectory_eval_fn(
            self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
            )

        self.optimizer.reset()


    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        env_params = None
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary
            if self.action_count == 0:
                def trajectory_eval_fn(env_params, action_sequences):
                    # take the environmental parameters out of the action sequence and stich them to the initial observation for the dynamics model.

                    return self.trajectory_eval_fn(env_params, obs, action_sequences)

                start_time = time.time()
                env_params, plan = self.first_optimizer.optimize(trajectory_eval_fn)
                plan_time = time.time() - start_time

                self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
            else:
                def trajectory_eval_fn(action_sequences):
                    # take the environmental parameters out of the action sequence and stich them to the initial observation for the dynamics model.

                    return self.trajectory_eval_fn(obs, action_sequences)

                start_time = time.time()
                plan = self.optimizer.optimize(trajectory_eval_fn)
                plan_time = time.time() - start_time

                self.actions_to_use.extend([a for a in plan[: self.replan_freq]])

        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")

        self.action_count += 1
        return env_params, action

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences)

        plan = self.optimizer.optimize(trajectory_eval_fn)
        return plan



class MarksCEMOptimizer(Optimizer):
    """Implements the Cross-Entropy Method optimization algorithm.

    A good description of CEM [1] can be found at https://arxiv.org/pdf/2008.06389.pdf. This
    code implements the version described in Section 2.1, labeled CEM_PETS
    (but note that the shift-initialization between planning time steps is handled outside of
    this class by TrajectoryOptimizer).

    This implementation also returns the best solution found as opposed
    to the mean of the last generation.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.

    [1] R. Rubinstein and W. Davidson. "The cross-entropy method for combinatorial and continuous
    optimization". Methodology and Computing in Applied Probability, 1999.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[float],
        upper_bound: Sequence[float],
        param_lb: Sequence[float],
        param_ub: Sequence[float],
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        # currently breaks here, from mis-processing the arrays. Could switch to add the
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.param_lb = torch.tensor(param_lb, device=device, dtype=torch.float32)
        self.param_ub = torch.tensor(param_ub, device=device, dtype=torch.float32)
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.initial_param_var = ((self.param_ub - self.param_lb) ** 2) / 16
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        p0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> (torch.Tensor, torch.Tensor):
        """Runs the optimization using CEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor): the best solution found.
        """
        mu = x0.clone()
        mewtwo = p0.clone()
        var = self.initial_var.clone()
        p_var = self.initial_param_var.clone()

        best_solution = torch.empty_like(mu)
        best_params = torch.empty_like(mewtwo)
        best_value = -np.inf
        population = torch.zeros((self.population_size,) + x0.shape ).to(
            device=self.device
        )
        param_population = torch.zeros((self.population_size,) + p0.shape).to(device=self.device)

        for i in range(self.num_iterations):
            lb_dist = mu - self.lower_bound
            ub_dist = self.upper_bound - mu
            p_lb_dist = mewtwo - self.param_lb
            p_ub_dist = mewtwo - self.param_ub
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            p_mv = torch.min(torch.square(p_lb_dist/2), torch.square(p_ub_dist / 2))
            constrained_var = torch.min(mv, var)
            p_constrained = torch.min(p_mv, p_var)

            population = mbrl.util.math.truncated_normal_(population)
            population = population * torch.sqrt(constrained_var) + mu
            param_population = mbrl.util.math.truncated_normal_(param_population) * torch.sqrt(p_constrained) + mewtwo

            # these should have the same value for each particle..E.g. the returned tensors should look the same?
            values = obj_fun(param_population, population)

            if callback is not None:
                callback(population, values, i)

            # filter out NaN values
            values[values.isnan()] = -1e-10

            best_values, elite_idx = values.topk(self.elite_num)
            elite = population[elite_idx]
            elite_params = param_population[elite_idx]

            new_mu = torch.mean(elite, dim=0)
            new_mewtwo = torch.mean(elite_params, dim=0)
            new_var = torch.var(elite, unbiased=False, dim=0)
            new_p_var = torch.var(elite_params, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            mewtwo = self.alpha * mewtwo + (1 - self.alpha) * new_mewtwo
            p_var = self.alpha * p_var + (1 - self.alpha) * new_p_var

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()
                best_params = param_population[elite_idx[0]].clone()

        return (mewtwo, mu) if self.return_mean_elites else (best_params, best_solution)





######
def create_mark_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
) -> MarksOptimizerAgent:
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)

    agent = hydra.utils.instantiate(agent_cfg)

    #boom
    def trajectory_eval_fn(initial_state, action_sequences):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles
        )

    def env_trajectory_eval_fn(env_params, obs, action_sequences):
        init_states = []
        for conditions in env_params:
            initial_state = conditions.tolist() + obs[len(conditions):].tolist()
            init_states.append(initial_state)
        return model_env.evaluate_parameterized_action_sequences(action_sequences, initial_states=torch.Tensor(init_states), num_particles=num_particles)

    def flex_eval_func(*args):
        if len([*args]) == 3:
            return env_trajectory_eval_fn(*args)
        elif len([*args]) == 2:
            return trajectory_eval_fn(*args)
        else:
            raise Exception("Wrong Num args")
    agent.set_trajectory_eval_fn(flex_eval_func)
    return agent
