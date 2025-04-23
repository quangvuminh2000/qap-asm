import numpy as np, random, sys, os


def read_instance(file_path):
    with open(file_path, "r") as file:
        return np.array([int(elem) for elem in file.read().split()])


def delta_swap(F, D, p, i, j):
    """O(n) Taillard's delta swap"""
    if i == j:
        return 0
    n = len(p)
    pi, pj = p[i], p[j]
    d = (F[i, i] - F[j, j])*(D[pj, pj] - D[pi, pi]) +\
        (F[i, j] - F[j, i])*(D[pj, pi] - D[pi, pj])
    for k in range(n):
        if k in [i, j]: # skip the swap
            continue
        pk = p[k]
        d += (F[i, k] - F[j, k])*(D[pk, pj] - D[pi, pk]) +\
            (F[k, i] - F[k, j])*(D[pj, pk] - D[pi, pk])
    return d

def first_improve(F, D, p):
    """One pass of the first improvement strategy"""
    n = len(p)
    I = list(range(n))
    random.shuffle(I)
    for i in I:
        J = list(range(n))
        random.shuffle(J)
        for j in J:
            if i == j:
                continue
            d = delta_swap(F, D, p, i, j)
            if d < 0:                       # improving move found
                p[i], p[j] = p[j], p[i]     # apply swap in-place
                return True
    return False

def two_opt_qap(F, D, p, max_passes=2):
    "Twice the neighborhood scan"
    for _ in range(max_passes):
        if not first_improve(F, D, p):
            break
    return p

def qap_cost(F, D, p):
    """Objective f(π) = Σ_i Σ_j a_ij * b_{π(i)π(j)}"""
    idx = np.ix_(p, p)
    return (F * D[idx]).sum()


def has_qa(F, D, n_ants=10, iters=100, R_ratio=1/3,
            q_exploit=0.9, alpha_evap=0.1, Q=100):
    
    n = F.shape[0]
    R = max(1, int(R_ratio * n))
    # initial solutions
    colony = [np.random.permutation(n).tolist() for _ in range(n_ants)]
    for p in colony:
        two_opt_qap(F, D, p)            # initial improvement
    best_p = min(colony, key=lambda p: qap_cost(F, D, p))
    best_f = qap_cost(F, D, best_p)

    # pheromone matrix - τ_ij initialised to τ0 = 1/(Q·f(π*))
    tau0 = 1 / (Q * best_f)
    tau = np.full((n, n), tau0)

    intensify = True         # flag to intensify search
    stale = 0                # stagnation counter
    S = n // 2               # number of best solutions to remember

    # Main loop
    for it in range(iters):
        improved_this_iter = False
        # each ant modifies its own permutation
        for k in range(n_ants):
            p = colony[k].copy()
            # R pheromone-driven swaps (exploitation vs exploration)
            for _ in range(R):
                r = random.randrange(n)
                if random.random() < q_exploit:
                    # exploitation - pick s that maximizes τ_{r,π[s]} + τ_{s,π[r]}
                    s_candidates = list(range(n))
                    s_candidates.remove(r)
                    s = max(s_candidates, key=lambda s: tau[r, p[s]] + tau[s, p[r]])
                else:
                    # probabilistic exploration - pick s from uniform distribution
                    probs = np.array([tau[r, p[s]] + tau[s, p[r]]
                                      if s != r else 0 for s in range(n)],
                                      dtype=np.float64)
                    probs /= probs.sum()
                    s = random.choices(range(n), weights=probs)[0]
                # perform the swap
                p[r], p[s] = p[s], p[r]
            
            # local search (two passes)
            two_opt_qap(F, D, p)

            # save result
            colony[k] = p
            f_p = qap_cost(F, D, p)
            if f_p < best_f:        # global best found
                best_f, best_p = f_p, p.copy()
                improved_this_iter = True
                intensify = True    # (re)activate intensification
                tau0 = 1 / (Q * best_f)
        
        # intensification
        if intensify:
            # keep the best between old and new for each ant
            for k in range(n_ants):
                old_f = qap_cost(F, D, colony[k])
                new_p = colony[k].copy()
                two_opt_qap(F, D, new_p)
                new_f = qap_cost(F, D, new_p)
                if new_f < old_f:
                    colony[k] = new_p
            # deactivate if no improvement
            if not improved_this_iter:
                intensify = False

        # update pheromone matrix
        tau *= (1 - alpha_evap)     # evaporation
        for i, j in enumerate(best_p): # reinforcement
            tau[i, j] += alpha_evap / best_f
        
        # diversification
        stale = 0 if improved_this_iter else stale + 1
        if stale >= S:
            tau.fill(tau0)              # erase trails

            for k in range(1, n_ants):  # keep best ant, randomize the rest
                colony[k] = np.random.permutation(n).tolist()
            stale = 0
            intensify = False
    
    return best_p, best_f


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
        F = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
        D = np.array([[next(file_it) for _ in range(n)] for _ in range(n)])
        best_p, best_f = has_qa(F, D)
        # print("\tBest assignment: {}".format(best_p))
        print("\tBest cost: {}".format(best_f))
        if BKS == 0:
            if best_f == 0:
                print("\tGap: 0.00%")
            else:
                print("\tGap: inf%")
        else:
            print("\tGap: {:.2f}%".format(abs(best_f - BKS) / BKS * 100))
