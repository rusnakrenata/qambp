"""Calibration stage: sweep candidate penalty multipliers and record results.

For every generated graph this script runs PyMetis and Kernighan-Lin once,
then solves the QUBO formulation for each candidate multiplier in the grid
and stores every run in the database. The recorded (lambda_min, lambda_max)
intervals are what `gbr_prediction.train_gbr()` later learns from.

Usage:
    python main.py --mode hybrid --scope 100
    python main.py --mode qpu --scope 70
"""

import argparse
import random

from database_tables import (
    Base,
    engine,
    sessionmaker,
    DESCRIPTION_CALIBRATION,
    DESCRIPTION_QPU,
)
from graph_generation import generate_graph
from pymetis_testing import run_pymetis_partition
from kerninghan_lin_testing import run_kerninghan_lin
from store_into_db import store_graph_to_db, store_partition_to_db

# Edge probabilities for the Erdos-Renyi generator (Section 3.1 of the paper).
EDGE_PROBABILITIES = [0.1, 0.25, 0.5, 0.75]

# Graph sizes used for the hybrid-solver calibration experiments.
HYBRID_NODE_SIZES = [
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1500, 2000, 2500, 3000, 4000,
]

# Candidate penalty multipliers per graph size (Table 3 of the paper).
# lambda = lambda_est * lambda_mult, see Eq. (27).
LAMBDA_MULT_GRID = {
    100: [0.05, 0.1, 0.2, 0.4],
    200: [0.05, 0.1, 0.2, 0.4],
    300: [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    400: [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    500: [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    600: [0.005, 0.01, 0.03, 0.05, 0.1],
    700: [0.005, 0.01, 0.03, 0.05, 0.1],
    800: [0.005, 0.01, 0.03, 0.05, 0.1],
    900: [0.005, 0.01, 0.03, 0.05, 0.1],
    1000: [0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
    1500: [0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
    2000: [0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
    2500: [0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
    3000: [0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
    4000: [0.0005, 0.001, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1],
}

# Direct-QPU experiments are limited to small graphs by embedding overhead
# (Section 6.2). A single wider grid is used for all of these sizes.
QPU_NODE_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
QPU_LAMBDA_MULTS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 1.5, 2]

NUM_READS_SAMPLED = 10  # used by the QPU and the classical 'test' sampler


def main(scope=100, mode="hybrid", seed=None):
    """Run the calibration sweep.

    Parameters:
        scope: Number of graphs to generate.
        mode: 'hybrid' (QA HS, n from 100 to 4000) or 'qpu' (n <= 100).
        seed: Optional seed for reproducible graph selection.
    """
    if seed is not None:
        random.seed(seed)

    if mode == "qpu":
        node_sizes = QPU_NODE_SIZES
    elif mode == "hybrid":
        node_sizes = HYBRID_NODE_SIZES
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'hybrid' or 'qpu'.")

    # Create all tables
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)

    for i in range(scope):
        with Session() as session:
            print(f"Graph number: {i}")

            nr_of_nodes = random.choice(node_sizes)
            # NOTE: assign to a fresh name; overwriting the source list here
            # would break every iteration after the first.
            edge_probability = random.choice(EDGE_PROBABILITIES)

            graph_data = generate_graph(nr_of_nodes, edge_probability)
            G = graph_data["graph"]

            pymetis_data = run_pymetis_partition(G)
            kl_data = run_kerninghan_lin(G)

            db_graph = store_graph_to_db(
                session,
                graph_data,
                pymetis_data,
                kl_data,
                i=mode,
                # Tag matches the values the queries in sql/ filter on.
                description=(
                    DESCRIPTION_QPU if mode == "qpu" else DESCRIPTION_CALIBRATION
                ),
            )

            if mode == "qpu":
                lambda_mults = QPU_LAMBDA_MULTS
                num_reads = NUM_READS_SAMPLED
            else:
                lambda_mults = LAMBDA_MULT_GRID[nr_of_nodes]
                num_reads = 1  # the hybrid solver returns a single sample

            for lambda_mult in lambda_mults:
                store_partition_to_db(
                    session=session,
                    G=G,
                    db_graph=db_graph,
                    lambda_mult=lambda_mult,
                    lambda_total=lambda_mult * db_graph.lambda_est,
                    comp_type=mode,
                    num_reads=num_reads,
                )

            del db_graph


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["hybrid", "qpu"],
        default="hybrid",
        help="Solver to calibrate against (default: hybrid).",
    )
    parser.add_argument(
        "--scope",
        type=int,
        default=100,
        help="Number of graphs to generate (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible graph selection.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(scope=args.scope, mode=args.mode, seed=args.seed)
