# Quadratic Assignment Problem (QAP) Solvers

This repository contains multiple implementations of solvers for the Quadratic Assignment Problem (QAP) using different optimization techniques.

## Directory Structure

- `scripts/acs/`: Contains Ant Colony System (ACS) based solvers
- `scripts/hexaly/`: Contains a solver using the Hexaly optimization platform
- `scripts/ga/`: Contains a solver using the Genetic Algorithm (GA)

## Requirements

- Python 3.x
- NumPy
- pandas
- tqdm
- Hexaly Optimizer (for hexaly solver)

## Available Solvers

### 1. Ant Colony System (ACS) Solvers

#### a. ACS-QA (`acs_qa.py`)

This implementation uses Ant Colony Optimization to solve QAP problems.

**Usage:**

```bash
python scripts/heuristics/acs/acs_qa.py <instance_directory>
```

**Parameters:**

- `colony_size`: Number of ants (default: 10)
- `iterations`: Number of iterations (default: 100)
- `rho`: Pheromone evaporation rate (default: 0.1)
- `alpha`: Pheromone importance (default: 1)
- `beta`: Heuristic information importance (default: 2)
- `q0`: Exploitation probability (default: 0.9)

#### b. HAS-QA (`has_qa.py`)

Hybrid Ant System implementation for QAP with local search improvements.

**Usage:**

```bash
python scripts/heuristics/acs/has_qa.py <instance_directory>
```

**Parameters:**

- `n_ants`: Number of ants (default: 10)
- `iters`: Number of iterations (default: 100)
- `R_ratio`: Ratio for candidate list (default: 1/3)
- `q_exploit`: Exploitation probability (default: 0.9)
- `alpha_evap`: Pheromone evaporation rate (default: 0.1)
- `Q`: Pheromone deposit constant (default: 100)

### 2. Hexaly Solver (`hexaly/qap.py`)

Commercial optimization solver implementation using the Hexaly platform.

**Usage:**

```bash
python scripts/heuristics/hexaly/qap.py <instance_directory> [output_file] [time_limit]
```

**Parameters:**

- `instance_directory`: Directory containing the instance files
- `output_file`: (Optional) Path to save the solution
- `time_limit`: (Optional) Time limit in seconds (default: 5)

### 3. Genetic Algorithm (GA) Solver (`ga_qa.py`)

**Usage:**

```bash
python scripts/heuristics/ga/ga_qa.py <instance_directory>
```

**Parameters:**
You can change the parameters in the `scripts/ga/config.py` file.

- `POPULATION_SIZE`: Number of individuals in the population (default: 10)
- `CROSSOVER_PROBABILITY`: Probability of crossover (default: 0.9)
- `MUTATION_PROBABILITY`: Probability of mutation (default: 0.1)
- `NUMBER_OF_GENERATIONS`: Number of generations (default: 200)

## Experiment

You can run experiment for specific solver and the solution is added into the `experiment` directory in the root folder.

Here is the bash parameters for running the experiment

```bash
python scripts/run_experiments.py --help
```

Output:

```txt
usage: run_experiments.py [-h] [-alg {hybrid,acs,ga,hexaly}] [-i INSTANCE_DIR]

Run experiments on QAP algorithms

optional arguments:
  -h, --help            show this help message and exit
  -alg {hybrid,acs,ga,hexaly}, --algorithm {hybrid,acs,ga,hexaly}
                        Algorithm to run: 'hybrid', 'acs', 'ga', or 'hexaly'
  -i INSTANCE_DIR, --instance_dir INSTANCE_DIR
                        Directory containing the instances to solve
```

## Input File Format

The input files should follow this format:

1. First line: Size of the problem (\(n\))
2. Next \(n\) lines each \(n\) elements: Distance matrix (\(n \times n\))
3. Next \(n\) lines each \(n\) elements: Flow matrix (\(n \times n\))

## Output Format

The solvers output:

1. The best solution found
2. The objective value (total cost)
3. The permutation representing the assignment
4. (Optional) Solution saved to a CSV file in the format:
   - 1st column: instance name
   - 2nd column: best known solution
   - 3rd column: current solver solution
   - 4th column: gap of current solution with best known solution
   - 5th column: the run time for the solver to solve the instance

## Notes

- The ACS, HAS and GA implementations are metaheuristic approaches that may find good solutions but not necessarily optimal ones
- The Hexaly solver is a commercial optimization tool that may provide better solutions but requires a license
- All implementations support parallel processing where applicable
- The solvers include various local search improvements and optimization techniques
- For ACS, HAS solvers, parameters are placed in `scripts/config.ini`
- For GA solvers, parameters are placed in `scripts/heuristics/ga/config.py`

## Example Usage

```bash
# Run ACS solver on all instances in a directory
python scripts/acs/acs_qa.py instances

# Run HAS solver on all instances in a directory
python scripts/acs/has_qa.py instances

# Run Hexaly solver on all instances in a directory
python scripts/hexaly/qap.py instances solution.txt 60

# Run GA solver on all instances in a directory
python scripts/ga/ga_qa.py instances

# Run experiment on specific solvers
python scripts/run_experiments.py -alg hybrid -i instances
```
