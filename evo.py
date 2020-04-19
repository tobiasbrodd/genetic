from getopt import getopt, GetoptError
import numpy as np
import random
import sys


def run(size=(100, 1), gens=100, prop=0.5):
    """Runs simulation."""

    pop = population(size=size)
    best = 0

    evolving = True
    gen = 0
    while evolving:
        best = progress(pop, best)
        pop = evolve(pop, prop=prop)
        gen += 1
        evolving = gen < gens
    progress(pop, best, last=True)


def progress(pop, prev, last=False):
    """Prints progress."""

    fit = fitness(pop)
    best = np.amax(fit)

    prev_out = f"{prev:.2f}"
    out = f"{best:.2f}"
    clear = " " * len(prev_out)
    print(f"\r{clear}", end="")

    end = "\n" if last else ""
    print(f"\r{out}", end=end)

    return best


def population(size=(100, 1)):
    """Creates a population."""

    pop = np.random.uniform(low=0, high=10, size=size)

    return pop


def evolve(pop, prop=0.5):
    """Evolves a population."""

    sel = select(pop, prop=prop)
    evo = crossover(sel)
    evo = mutate(evo)

    n_pop = pop.shape[0]
    n_evo = evo.shape[0]
    n_rest = n_pop - n_evo

    if n_rest <= 0:
        return evo

    fit = fitness(pop)
    idx = np.argsort(fit)[::-1][:n_rest]
    best = pop[idx, :]
    evol = np.concatenate((best, evo), axis=0)

    return evol


def select(pop, prop=0.5, cross=2):
    """Selects a subpopulation."""

    n = pop.shape[0]
    fit = fitness(pop)
    fit_sum = np.sum(fit)
    min_fit = np.amin(fit_sum)
    if min_fit < 0:
        fit_sum += np.abs(min_fit) * n
        fit += np.abs(min_fit)
    probs = fit / fit_sum

    rows = int(n * prop)
    idx_size = (rows, cross)
    idx = np.random.choice(n, size=idx_size, replace=True, p=probs)

    cols = pop.shape[1]
    sel_size = (rows, cols, cross)
    sel = np.zeros(sel_size)
    for col in range(cross):
        pop_idx = idx[:, col]
        sel[:, :, col] = pop[pop_idx, :]

    return sel


def fitness(pop):
    """Evaluates fitness for a population."""

    return np.sum(pop, axis=1)


def crossover(sel):
    """Crosses a population."""

    pop = np.mean(sel, axis=2)

    return pop


def mutate(pop):
    """Mutates a population."""

    size = pop.shape
    pop += np.random.uniform(low=-1, high=1, size=size)

    return pop


def main(argv):
    short_options = "h"
    long_options = ["help", "seed=", "size=", "gens=", "prop="]
    help_message = """usage: evo.py [options]
    options:
        -h, --help          Prints help message.
        --seed s            Sets seed 's'. Default: '123'.
        --size s            Sets population size 's'. Default: '100'.
        --gens g            Sets generations 'g'. Default: '100'.
        --prop p            Sets proportion 'p'. Default: '0.5'."""

    try:
        opts, args = getopt(argv, shortopts=short_options, longopts=long_options)
    except GetoptError:
        print(help_message)
        return

    seed = 123
    n = 100
    gens = 100
    prop = 0.5

    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(helsp_message)
            return
        elif opt == "--seed":
            seed = int(arg)
        elif opt == "--size":
            n = int(arg)
        elif opt == "--gens":
            gens = int(arg)
        elif opt == "--prop":
            prop = float(arg)

    random.seed(seed)
    np.random.seed(seed)

    run(size=(n, 1), gens=gens, prop=prop)


if __name__ == "__main__":
    main(sys.argv[1:])
