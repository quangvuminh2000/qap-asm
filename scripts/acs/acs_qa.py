import sys
import os
import numpy as np


def ant_colony_optimization(
    F,
    D,
    B=None,
    colony_size=10,
    iterations=100,
    rho=0.1,
    alpha=1,
    beta=2,
    q0=0.9,
    use_cost_matrix=True,
    debug=False,
):
    n = len(F)
    pheromone = np.ones((n, n)) / n
    best_cost = float("inf")
    best_assignment = None

    def generate_solutions(colony_size, pheromone, F, D, B, alpha, beta, q0):
        solutions = []
        for _ in range(colony_size):
            solution = construct_solution(pheromone, F, D, B, alpha, beta, q0)
            solutions.append(solution)
        return solutions

    def construct_solution(pheromone, F, D, B, alpha, beta, q0):
        n = len(F)
        solution = np.zeros((n, n))
        unassigned_facilities = set(range(n))
        for i in range(n):
            current_location = i
            current_facility = choose_next_facility(
                pheromone,
                F,
                D,
                B,
                current_location,
                unassigned_facilities,
                alpha,
                beta,
                q0,
            )
            solution[i, current_facility] = 1
            unassigned_facilities.remove(current_facility)
        return solution

    def choose_next_facility(
        pheromone, F, D, B, current_location, unassigned_facilities, alpha, beta, q0
    ):
        n = len(F)
        if np.random.rand() < q0:  # Use probabilities to select the next facility
            probabilities = calculate_probabilities(
                pheromone, F, D, B, current_location, unassigned_facilities, alpha, beta
            )
            try:
                selected_facility = np.random.choice(
                    list(unassigned_facilities), p=probabilities
                )
            except:
                print(probabilities)
                print(unassigned_facilities)
                print(current_location)
                raise
        else:  # greedy choice based on pheromone
            probabilities = (
                pheromone[current_location, list(unassigned_facilities)] ** alpha
            )
            selected_facility = list(unassigned_facilities)[np.argmax(probabilities)]
        return selected_facility

    def calculate_probabilities(
        pheromone, F, D, B, current_location, unassigned_facilities, alpha, beta
    ):
        probabilities = np.zeros(len(unassigned_facilities))
        for i, facility in enumerate(unassigned_facilities):
            pheromone_factor = pheromone[current_location, facility] ** alpha
            if use_cost_matrix:
                weight_distance_factor = (
                    np.sum(F * D[:, [facility]] + B, axis=1) ** beta
                )
            else:
                weight_distance_factor = np.sum(F @ D[:, [facility]], axis=1) ** beta
            max_weight_distance_factor = weight_distance_factor.max()
            if max_weight_distance_factor == 0:
                probabilities[i] = 0
            else:
                probabilities[i] = pheromone_factor / (max_weight_distance_factor)
            if np.isnan(probabilities[i]):
                print(probabilities[i])
                print(pheromone_factor)
                print(max_weight_distance_factor)
                raise
        probabilities /= probabilities.sum()
        return probabilities

    def pheromone_update(pheromone, solutions, rho):
        pheromone *= 1 - rho
        for solution, cost in solutions:
            pheromone += solution / cost

    temp_cost = best_cost
    for i in range(iterations):
        solutions = generate_solutions(colony_size, pheromone, F, D, B, alpha, beta, q0)
        for solution in solutions:
            if use_cost_matrix:
                cost = np.trace(F @ solution @ D @ solution.T - 2*B @ solution.T)
            else:
                cost = np.trace(F @ solution @ D @ solution.T)

            if cost < best_cost:
                best_cost = cost
                best_assignment = solution
        if use_cost_matrix:
            pheromone_update(
                pheromone,
                [
                    (
                        solution,
                        np.trace(F @ solution @ D @ solution.T - 2*B @ solution.T)
                    )
                    for solution in solutions
                ],
                rho,
            )
        else:
            pheromone_update(
                pheromone,
                [
                    (solution, np.trace(F @ solution @ D @ solution.T))
                    for solution in solutions
                ],
                rho,
            )
        if debug and temp_cost > best_cost:
            print(
                "Iter {}\n\tprev cost:{}\n\tcurr cost: {}".format(
                    i, temp_cost, best_cost
                )
            )
            temp_cost = best_cost

    return best_assignment, best_cost


def read_instance(file_path):
    with open(file_path, "r") as file:
        return np.array([int(elem) for elem in file.read().split()])


if __name__ == "__main__":
    instance_dir = sys.argv[1]
    for instance_name in os.listdir(instance_dir):
        print("Current data: {}".format(instance_name))
        instance_path = os.path.join(instance_dir, instance_name)
        instance_file = os.path.join(instance_path, "{}.txt".format(instance_name))
        solution_file = os.path.join(instance_path, "solution.txt")
        
        if not os.path.exists(instance_file):
            print("\tInstance file not found")
            continue
        if not os.path.exists(solution_file):
            print("\tSolution file not found")
            continue
        
        file_it = iter(read_instance(instance_file))
        soln_it = iter(read_instance(solution_file)[1:])
        n = next(file_it)
        BKS = next(soln_it)

        np.random.seed(2025)
        F = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
        D = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
        best_assignment, best_cost = ant_colony_optimization(
            F, D, iterations=100, debug=False, use_cost_matrix=False
        )
        print("\tBest cost: {}".format(best_cost))
        if BKS == 0:
            if best_cost == 0:
                print("\tGap: 0.00%")
            else:
                print("\tGap: 100.00%")
        else:
            print("\tGap: {:.2f}%".format(abs(best_cost - BKS) / BKS * 100))
