import hexaly.optimizer
import sys, os

if len(sys.argv) < 2:
    print("Usage: python qap.py inputFile [outputFile] [timeLimit]")
    sys.exit(1)


def read_integers(filename):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]


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
    file_it = iter(read_integers(instance_file))
    soln_it = iter(read_integers(solution_file)[1:])
    n = next(file_it)
    BKS = next(soln_it)

    # Distance between locations
    A = [[next(file_it) for j in range(n)] for i in range(n)]
    # Flow between factories
    B = [[next(file_it) for j in range(n)] for i in range(n)]

    #
    # Declare the optimization model
    #
    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        model = optimizer.model

        # Permutation such that p[i] is the facility on the location i
        p = model.list(n)

        # The list must be complete
        model.constraint(model.eq(model.count(p), n))

        # Create B as an array to be accessed by an at operator
        array_B = model.array(B)

        # Minimize the sum of product distance*flow
        obj = model.sum(A[i][j] * model.at(array_B, p[i], p[j])
                        for j in range(n) for i in range(n))
        model.minimize(obj)

        model.close()

        # Parameterize the optimizer
        optimizer.param.verbosity = 0
        if len(sys.argv) >= 4:
            optimizer.param.time_limit = int(sys.argv[3])
        else:
            optimizer.param.time_limit = 5
        optimizer.solve()
        print("\tBest cost: {}".format(obj.value))
        if BKS == 0:
            if obj.value == 0:
                print("\tGap: 0.00%")
            else:
                print("\tGap: 100.00%")
        else:
            print("\tGap: {:.2f}%".format(abs(obj.value - BKS) / BKS * 100))

        #
        # Write the solution in a file with the following format:
        #  - n objValue
        #  - permutation p
        #
        if len(sys.argv) >= 3:
            with open(sys.argv[2], 'w') as outfile:
                outfile.write("%d %d\n" % (n, obj.value))
                for i in range(n):
                    outfile.write("%d " % p.value[i])
                outfile.write("\n")
