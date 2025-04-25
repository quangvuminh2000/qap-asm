import sys, os.path, time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ga.config import POPULATION_SIZE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY


def generate_random_population(n_objects, n_chromosomes):
    population_list = []
    for _ in range(n_chromosomes):
        rand_chromosome = list(range(n_objects))
        np.random.shuffle(rand_chromosome)
        population_list.append(rand_chromosome)
    return population_list


def main():
    population = generate_random_population(10, POPULATION_SIZE)
    print(population)


if __name__ == "__main__":
    main()