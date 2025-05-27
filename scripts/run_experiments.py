import os
from typing import Callable, Any, Tuple, Dict
from time import time
from datetime import datetime

import pandas as pd
import numpy as np

from utils import read_instance
from utils.instance_utils import load_config
from heuristics import has_qa, ant_colony_optimization, ga_qa
from hexaly_qap import hexaly_qap

from argparse import ArgumentParser


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
CONFIG = load_config(CONFIG_PATH)

ALGO_DICT: Dict[str, Callable] = {
    "hybrid": has_qa,
    "acs": ant_colony_optimization,
    "ga": ga_qa,
    "hexaly": hexaly_qap
}

CONFIG_DICT: Dict[str, Any] = {
    "hybrid": {k: float(v) for k, v in CONFIG["HAS"].items()},
    "acs": {k: float(v) for k, v in CONFIG["ACS"].items()},
}


def experiment(
    func: Callable, instance_dir: str, *args: Tuple[Any], **kwargs: Dict[str, Any]
) -> None:
    """
    Run an experiment with the given algorithm and parameters. Then save the results as .csv file.

    Parameters
    ----------
    func : Callable
        solver function to be run
    instance_dir : str
        directory containing the instances
    """

    if not os.path.exists(instance_dir):
        print(f"Instance directory {instance_dir} does not exist")
        return

    best_knowns = []
    prob_names = []
    curr_sols = []
    gaps = []
    times = []
    for instance_name in os.listdir(instance_dir):
        print("Current data: {}".format(instance_name))
        instance_path = os.path.join(instance_dir, instance_name)
        instance_file = os.path.join(instance_path, "{}.txt".format(instance_name))
        solution_file = os.path.join(instance_path, "solution.txt")

        # Check if the instance and solution files exist
        if not os.path.exists(instance_file):
            print("\tInstance file not found")
            continue

        file_it = iter(read_instance(instance_file))
        n = next(file_it)
        D = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
        F = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])

        if not os.path.exists(solution_file):
            print(
                "\tSolution file not found, setting the best known to inf and trying to solve"
            )
            BKS = float("inf")
        else:
            sol_it = iter(read_instance(solution_file)[1:])
            BKS = next(sol_it)

        # Run the algorithm
        start_time = time()
        best_p, best_f = func(F, D, *args, **kwargs)
        algo_time = time() - start_time

        print("\tBest cost: {}".format(best_f))
        gap = 0
        if BKS == 0:
            if best_f == 0:
                gap = 0
                print("\tGap: 0.00%")
            else:
                gap = float("inf")
                print("\tGap: inf%")
        elif BKS == float("inf"):
            gap = 0
            print("\tGap: 0.00%")
        else:
            gap = abs(best_f - BKS) / BKS * 100
            print("\tGap: {:.2f}%".format(gap))

        # Store the results
        prob_names.append(instance_name)
        curr_sols.append(best_f)
        best_knowns.append(BKS)
        gaps.append(gap)
        times.append(algo_time)

    # Create a dataframe to store results
    results = pd.DataFrame(
        {
            "instance": prob_names,
            "best_known": best_knowns,
            "algo_solution": curr_sols,
            "gap": gaps,
            "time": times,
        }
    )

    # Save the results to a CSV file
    experiment_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments"
    )
    os.makedirs(
        experiment_dir,
        exist_ok=True,
    )

    today = datetime.today().strftime("%Y-%m-%d_%H-%M")
    results.to_csv(
        os.path.join(experiment_dir, f"{func.__name__}_{today}_results.csv"),
        index=False,
        float_format="%.4f",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run experiments on QAP algorithms")
    parser.add_argument(
        "-alg",
        "--algorithm",
        choices=ALGO_DICT.keys(),
        type=str,
        help="Algorithm to run: 'hybrid', 'acs', 'ga' or 'hexaly'",
    )
    parser.add_argument(
        "-i", "--instance_dir", help="Directory containing the instances to solve"
    )
    args = parser.parse_args()

    print(f"Running experiments with algorithm: {args.algorithm}")
    print(f"Using instance directory: {args.instance_dir}")
    if args.algorithm in CONFIG_DICT:
        print(f"Algorithm configuration:")
        for k, v in CONFIG_DICT.get(args.algorithm, {}).items():
            print(f"\t{k}:{v}")

    algo_func = ALGO_DICT[args.algorithm]
    if args.algorithm in CONFIG_DICT:
        experiment(algo_func, args.instance_dir, **CONFIG_DICT[args.algorithm])
    else:
        experiment(algo_func, args.instance_dir)
