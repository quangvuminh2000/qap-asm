import sys, os.path
import numpy as np

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ga.config import POPULATION_SIZE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, NUMBER_OF_GENERATIONS
from ga.data_loading import load_square_matrix
from ga.fitnes_function import compute_fitness_scores_list, get_normalized_result_of_fitness_function_scores_list
from ga.generate_population import generate_random_population
from ga.selection import TournamentSelection, RouletteSelection, Selection
from ga.mutation import BasicMutation, Mutation
from ga.crossover import BasicCrossover, Crossover

selection_strategy = Selection(selection_algorithm=TournamentSelection())
mutation_strategy = Mutation(mutation_algorithm=BasicMutation(), mutation_probability=MUTATION_PROBABILITY)
crossover_strategy = Crossover(crossover_algorithm=BasicCrossover(), crossover_probability=CROSSOVER_PROBABILITY)



def main():

    instance_dir = sys.argv[1]
    for instance_name in os.listdir(instance_dir):
        print("Current data: {}".format(instance_name))
        try:
            n, F, D, BKS = load_square_matrix(instance_dir, instance_name)
        except Exception as e:
            print("\tError: {}".format(e))
            continue

        population = generate_random_population(n, POPULATION_SIZE)

        average_results = []
        min_results = []
        max_results = []

        for generation in tqdm(range(NUMBER_OF_GENERATIONS), desc="Generation", leave=False):

            fitness_scores = compute_fitness_scores_list(population, D, F)
            fitness_scores_normalized = get_normalized_result_of_fitness_function_scores_list(fitness_scores)

            # not normalized yet, max means "the worst", therefor "min" for us
            max_fitness = np.min(fitness_scores)
            min_fitness = np.max(fitness_scores)
            average_fitness = np.mean(fitness_scores)

            max_results.append(max_fitness)
            min_results.append(min_fitness)
            average_results.append(average_fitness)

            max_chromosome = population[np.argmin(fitness_scores)]
            max_chromosome = list(map(lambda x: x + 1, max_chromosome))

            selected_chromosomes = selection_strategy.select(population, fitness_scores_normalized)
            crossed_chromosomes = crossover_strategy.crossover(selected_chromosomes)
            mutated_chromosomes = mutation_strategy.mutate(crossed_chromosomes)

            population = mutated_chromosomes

        print("\tBest cost: {}".format(max_results[-1]))
        if BKS == 0:
            if max_results[-1] == 0:
                print("\tGap: 0.00%")
            else:
                print("\tGap: inf%")
        else:
            print("\tGap: {:.2f}%".format(abs(max_results[-1] - BKS) / BKS * 100))



if __name__ == "__main__":
    main()