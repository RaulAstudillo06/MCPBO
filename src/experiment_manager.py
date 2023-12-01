from typing import Callable, Dict, List, Optional

from src.one_trial import one_trial


def experiment_manager(
    problem: str,  # problem id
    utility_func: Callable,  # utility function (x -> f(x))
    input_dim: int,  # input dimension
    num_attributes: int,  # number of attributes (i.e., output dimension of f)
    obs_attributes: List,  #
    comp_noise_type: str,  # type of comparison noise ("probit" and "logit" noise are supported)
    comp_noise: float,  # scalar determining the magnitude of the comparison noise
    algo: str,  # algo or acquisition function id
    batch_size: int,  # number of items in the query (e.g., 2 for standard binary queries)
    num_init_queries: int,  # number of initial queries (these are selected uniformly at random over the input space)
    num_algo_iter: int,  # number of queries selected by the algorithm
    first_trial: int,  # first trial id to be ran
    last_trial: int,  # last trial id to be ran
    restart: bool,  # if True, this will try to restart the experiment from existing data
    ignore_failures: bool = False,  # ignore this for now
    algo_params: Optional[Dict] = None,  # ignore this for now
) -> None:
    # `trial` determines the random seed of each trial
    for trial in range(first_trial, last_trial + 1):
        one_trial(
            problem=problem,
            utility_func=utility_func,
            input_dim=input_dim,
            num_attributes=num_attributes,
            obs_attributes=obs_attributes,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            algo=algo,
            algo_params=algo_params,
            batch_size=batch_size,
            num_init_queries=num_init_queries,
            num_algo_iter=num_algo_iter,
            trial=trial,
            restart=restart,
            ignore_failures=ignore_failures,
        )
