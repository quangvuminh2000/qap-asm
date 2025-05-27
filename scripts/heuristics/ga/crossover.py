
import copy, random


class Crossover(object):

    def __init__(self, crossover_algorithm, crossover_probability: float):
        self.crossover_algorithm = crossover_algorithm
        self.crossover_probability = crossover_probability

    def crossover(self, population: list[list[int]]) -> list[list[int]]:
        return self.crossover_algorithm(population, self.crossover_probability)


class BasicCrossover(object):

    def __init__(self):
        pass

    def __call__(self, population: list[list[int]], crossover_probability: float) -> list[list[int]]:
        return self.basic_crossover(population, crossover_probability)

    @staticmethod
    def basic_crossover(population: list[list[int]], crossover_probability: float) -> list[list[int]]:
        species_not_crossovered = []
        species_to_crossover = []

        BasicCrossover.choose_chromosomes_to_crossover(population, species_not_crossovered, species_to_crossover, crossover_probability)

        crossover_tuples = BasicCrossover.create_crossover_tuples(species_not_crossovered, species_to_crossover)

        crossovered_species = BasicCrossover.crossover_population(crossover_tuples)

        return crossovered_species + species_not_crossovered

    @staticmethod
    def crossover_population(crossover_tuples):
        crossovered_species = []
        for crossover_tuple in crossover_tuples:
            child_a, child_b = BasicCrossover.crossover_chromosomes(
                crossover_tuple,
                point_of_crossover=random.randint(0, len(crossover_tuple) - 1)
            )
            crossovered_species.append(child_a)
            crossovered_species.append(child_b)
        return crossovered_species

    @staticmethod
    def choose_chromosomes_to_crossover(population, species_not_crossovered, species_to_crossover, crossover_probability: float):
        for chromosome in population:
            if random.uniform(0, 1) < crossover_probability:
                species_to_crossover.append(chromosome)
            else:
                species_not_crossovered.append(chromosome)

    @staticmethod
    def create_crossover_tuples(species_not_crossovered, species_to_crossover):
        crossover_tuples = []
        species_to_crossover = list(enumerate(species_to_crossover))
        while species_to_crossover:
            chromosome_to_crossover_index, chromosome_to_crossover = species_to_crossover.pop()

            if not species_to_crossover:
                species_not_crossovered.append(chromosome_to_crossover)
                break

            crossover_buddy_index, crossover_buddy = random.choice(species_to_crossover)
            species_to_crossover = list(filter(lambda value: value[0] != crossover_buddy_index, species_to_crossover))
            crossover_tuples.append((chromosome_to_crossover, crossover_buddy))
        return crossover_tuples

    @staticmethod
    def crossover_chromosomes(parents, point_of_crossover):
        father, mother = parents
        child_a, child_b = copy.copy(father), copy.copy(mother)
        # Change chromosome
        for index in range(point_of_crossover):
            # Values on index
            value_a = child_a[index]
            value_b = child_b[index]
            # Values for swap
            index_of_value_b_in_a = child_a.index(value_b)
            index_of_value_a_in_b = child_b.index(value_a)
            # Save values
            child_a[index_of_value_b_in_a] = value_a
            child_b[index_of_value_a_in_b] = value_b
            # Change values
            child_a[index] = value_b
            child_b[index] = value_a
        return child_a, child_b
                

