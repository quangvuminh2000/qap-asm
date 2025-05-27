import numpy as np

def compute_fitness_scores_list(
        population: list[list[int]],
          distance_matrix: np.ndarray,
            flow_matrix: np.ndarray) -> list[int]:

    matrices_size = distance_matrix.shape[0]
    fitness_scores_list = []
    for chromosome in population:
        assert len(chromosome) == len(set(chromosome))
        chromosome_fitness_sum = 0
        for x in range(matrices_size):
            for y in range(matrices_size):
                chromosome_fitness_sum += distance_matrix[x,y] * flow_matrix[chromosome[x]-1, chromosome[y]-1]
        fitness_scores_list.append(chromosome_fitness_sum)
    return fitness_scores_list


def get_normalized_result_of_fitness_function_scores_list(
        fitness_scores_list: list[int]) -> list[float]:

    map_to_minimization_problem = list(map(lambda value: 1. / value, fitness_scores_list))
    normalized_results = np.array(map_to_minimization_problem) / np.sum(map_to_minimization_problem)
    return normalized_results
