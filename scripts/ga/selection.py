import numpy as np
import random


class Selection(object):

    def __init__(self, selection_algorithm):
        self.selection_algorithm = selection_algorithm

    def select(self, population: list[list[int]], fitness_scores: list[float]) -> list[list[int]]:
        return self.selection_algorithm(population, fitness_scores)


class TournamentSelection(object):

    def __init__(self):
        pass

    def __call__(self, population: list[list[int]], fitness_scores: list[float]) -> list[list[int]]:
        return self.tournament_selection(population, fitness_scores)

    @staticmethod
    def tournament_selection(population: list[list[int]], fitness_scores: list[float], elitism: bool = False) -> list[list[int]]:
        new_species = []
        population_size = len(fitness_scores)
        population_size = population_size - 1 if elitism else population_size
        for _ in range(population_size):
            # take best
            of_parent_idx = random.randint(0, len(fitness_scores) - 1)
            tf_parent_idx = random.randint(0, len(fitness_scores) - 1)
            if fitness_scores[of_parent_idx] > fitness_scores[tf_parent_idx]:
                ch_winner = population[of_parent_idx]
            else:
                ch_winner = population[tf_parent_idx]
            new_species.append(ch_winner)
        return new_species


class RouletteSelection(object):

    def __init__(self):
        pass

    def __call__(self, population: list[list[int]], fitness_scores: list[float]) -> list[list[int]]:
        return self.roulette_selection(population, fitness_scores)

    @staticmethod
    def roulette_selection(population: list[list[int]], fitness_scores: list[float]) -> list[list[int]]:
        new_population = []

        # remove maximum value from every element so roulette selection will rely on bigger difference
        worst_result = np.min(fitness_scores)
        fitness_scores = list(map(lambda x: x - worst_result, fitness_scores))

        cummulative_sum = np.cumsum(fitness_scores)

        for _ in range(len(population)):
            probability_of_choose = random.uniform(0, 1) * sum(fitness_scores)
            randomly_selected_member = RouletteSelection.select_chromosome(population, cummulative_sum, probability_of_choose)

            new_population.append(randomly_selected_member)

        return new_population


    @staticmethod
    def select_chromosome(population: list[list[int]], cummulative_sum: list[float], probability_of_choose: float) -> list[int]:
        for index, _ in enumerate(cummulative_sum):
            if probability_of_choose <= cummulative_sum[index]:
                return population[index]


