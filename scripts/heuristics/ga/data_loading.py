import os
import numpy as np
from typing import Tuple, Optional


def read_instance(file_path):
    with open(file_path, "r") as file:
        return np.array([int(elem) for elem in file.read().split()])


def load_square_matrix(instance_dir: str, instance_name: str) -> Optional[Tuple[int, np.ndarray, np.ndarray, int]]:
    instance_path = os.path.join(instance_dir, instance_name)
    instance_file = os.path.join(instance_path, "{}.txt".format(instance_name))
    solution_file = os.path.join(instance_path, "solution.txt")
    if not os.path.exists(instance_path) or not os.path.exists(solution_file):
        raise Exception(f"File not found: {instance_path} or {solution_file}")
    
    file_it = iter(read_instance(instance_file))
    soln_it = iter(read_instance(solution_file)[1:])
    n = next(file_it)
    BKS = next(soln_it)
    F = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
    D = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
    return n, F, D, BKS
    


