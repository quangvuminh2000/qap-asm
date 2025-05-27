import random

def generate_random_population(n_objects: int, n_chromosomes: int) -> list[list[int]]:
    population_list = []
    for _ in range(n_chromosomes):
        rand_chromosome = list(range(n_objects))
        random.shuffle(rand_chromosome)
        population_list.append(rand_chromosome)
    return population_list