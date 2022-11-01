from typing import Callable, Dict, List, Optional

import os
import sys

from src.pbo_trial import pbo_trial


def experiment_manager(
    problem: str,
    obj_func: Callable,
    utility_func,
    input_dim: int,
    output_dim: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    first_trial: int,
    last_trial: int,
    restart: bool,
    model_type: str = "Standard",
    add_baseline_point: bool = False,
    ignore_failures: bool = False,
    algo_params: Optional[Dict] = None,
) -> None:

    for trial in range(first_trial, last_trial + 1):
        pbo_trial(
            problem=problem,
            obj_func=obj_func,
            utility_func=utility_func,
            input_dim=input_dim,
            output_dim=output_dim,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            algo=algo,
            algo_params=algo_params,
            batch_size=batch_size,
            num_init_queries=num_init_queries,
            num_algo_iter=num_algo_iter,
            trial=trial,
            restart=restart,
            model_type=model_type,
            add_baseline_point=add_baseline_point,
            ignore_failures=ignore_failures,
        )