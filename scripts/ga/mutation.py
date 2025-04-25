import random


class Mutation(object):
    def __init__(self, mutation_algorithm, mutation_probability: float):
        self.mutation_algorithm = mutation_algorithm
        self.mutation_probability = mutation_probability

    def mutate(self, population: list[list[int]]) -> list[list[int]]:
        return self.mutation_algorithm(population, self.mutation_probability)


class BasicMutation(object):

    def __init__(self):
        pass

    def __call__(self, population: list[list[int]], mutation_probability: float) -> list[list[int]]:
        return self.basic_mutation(population, mutation_probability)

    @staticmethod
    def basic_mutation(population: list[list[int]], mutation_probability: float) -> list[list[int]]:
        chromosomes_with_mutated = []

        for chromosome in population:
            if 0 <= random.uniform(0, 1) <= mutation_probability:
                mutated_chromosome = BasicMutation.mutate_chromosome(
                    chromosome,
                    BasicMutation.generate_random_indices(len(chromosome))
                )
                chromosomes_with_mutated.append(mutated_chromosome)
            else:
                chromosomes_with_mutated.append(chromosome)

        return chromosomes_with_mutated

    @staticmethod
    def mutate_chromosome(chromosome: list[int], indices: list[int]) -> list[int]:
        gen_a_index, gen_b_index = indices
        chromosome[gen_a_index], chromosome[gen_b_index] = chromosome[gen_b_index], chromosome[gen_a_index]
        return chromosome

    @staticmethod
    def generate_random_indices(length: int) -> list[int]:
        return random.sample(range(length), 2)
