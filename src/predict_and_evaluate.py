"""Evaluation stage: solve the MBP using a GBR-predicted penalty parameter.

This is the counterpart to ``main.py``. Where ``main.py`` sweeps a grid of
candidate multipliers to build the calibration dataset, this script performs
Algorithm 3 of the paper: for each newly generated graph it predicts the
penalty parameter from the trained regressors, constructs the QUBO once, and
solves it once.

    1. Compute lambda_est = (1 + min(n/2 - 1, max(deg(G)))) / 2      Eq. (26)
    2. Build the feature vector f = (n, rho, lambda_est)
    3. Predict (lambda_min, lambda_max) with gbr_min / gbr_max
    4. lambda = lambda_est * (lambda_min + lambda_max) / 2           Eq. (28)

Results are stored in the same tables as the calibration runs and tagged with
a distinct ``description`` so the two stages can be told apart.

Usage:
    python predict_and_evaluate.py --mode hybrid --scope 126
    python predict_and_evaluate.py --mode qpu --scope 70

Requires trained models in ``src/gbr/``; run ``python gbr_prediction.py
--train`` first if they are absent.
"""

import argparse
import random
import statistics

from database_tables import (
    Base,
    engine,
    sessionmaker,
    DESCRIPTION_EVALUATION,
    DESCRIPTION_QPU,
)
from gbr_prediction import predict_lambda
from graph_generation import generate_graph
from pymetis_testing import run_pymetis_partition
from kerninghan_lin_testing import run_kerninghan_lin
from store_into_db import store_graph_to_db, store_partition_to_db
from qa_testing import INVALID_CUT

# Section 6.1: hybrid-solver evaluation covers 100 <= n <= 4000.
HYBRID_NODE_SIZES = [
    100, 200, 300, 400, 500, 600, 700, 800, 900,
    1000, 1500, 2000, 2500, 3000, 4000,
]

# Section 6.2: direct-QPU evaluation is limited to n <= 100 by embedding
# overhead on the Advantage (Pegasus) topology.
QPU_NODE_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

EDGE_PROBABILITIES = [0.1, 0.25, 0.5, 0.75]

NUM_READS_QPU = 10  # matches the annealing settings reported in Section 6.2


def evaluate(scope=126, mode="hybrid", seed=None):
    """Generate graphs, solve them with a predicted penalty, and store results.

    Parameters:
        scope: Number of graphs to evaluate (126 hybrid / 70 QPU in the paper).
        mode: 'hybrid' or 'qpu'.
        seed: Optional seed for reproducible graph selection.

    Returns:
        list of per-graph result dicts, also summarised to stdout.
    """
    if seed is not None:
        random.seed(seed)

    if mode == "qpu":
        node_sizes, num_reads = QPU_NODE_SIZES, NUM_READS_QPU
    elif mode == "hybrid":
        node_sizes, num_reads = HYBRID_NODE_SIZES, 1
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'hybrid' or 'qpu'.")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    results = []

    for i in range(scope):
        with Session() as session:
            nr_of_nodes = random.choice(node_sizes)
            edge_probability = random.choice(EDGE_PROBABILITIES)

            graph_data = generate_graph(nr_of_nodes, edge_probability)
            G = graph_data["graph"]

            # Classical baselines
            pymetis_data = run_pymetis_partition(G)
            kl_data = run_kerninghan_lin(G)

            # Algorithm 3: predict the penalty parameter for this graph
            prediction = predict_lambda(G)

            print(
                f"[{i + 1}/{scope}] n={nr_of_nodes} p={edge_probability} "
                f"lambda_est={prediction['lambda_est']:.3f} "
                f"mult={prediction['lambda_mult']:.4f} "
                f"lambda={prediction['lambda_total']:.4f}"
            )

            db_graph = store_graph_to_db(
                session,
                graph_data,
                pymetis_data,
                kl_data,
                i=f"{mode}_gbr",
                # Tag matches the values the queries in sql/ filter on.
                description=(
                    DESCRIPTION_QPU if mode == "qpu" else DESCRIPTION_EVALUATION
                ),
            )

            db_partition = store_partition_to_db(
                session=session,
                G=G,
                db_graph=db_graph,
                lambda_total=prediction["lambda_total"],
                lambda_mult=prediction["lambda_mult"],
                comp_type=mode,
                num_reads=num_reads,
            )
            qa_cut = db_partition.qa_inter_edges

            results.append(
                {
                    "nr_of_nodes": nr_of_nodes,
                    "edge_probability": edge_probability,
                    "density": graph_data["density"],
                    "lambda_total": prediction["lambda_total"],
                    "qa_inter_edges": qa_cut,
                    "pymetis_inter_edges": pymetis_data["pymetis_inter_edges"],
                    "kl_inter_edges": kl_data["kl_inter_edges"],
                }
            )

    summarise(results, mode)
    return results


def summarise(results, mode):
    """Print the comparison reported in Table 4 of the paper."""
    feasible = [r for r in results if r["qa_inter_edges"] < INVALID_CUT]

    print(f"\n--- {mode} evaluation summary ({len(results)} graphs) ---")

    if not results:
        return

    print(
        f"QA feasible (balanced) solutions: "
        f"{len(feasible)}/{len(results)} "
        f"({100 * len(feasible) / len(results):.1f}%)"
    )

    if not feasible:
        return

    wins = sum(
        1 for r in feasible if r["qa_inter_edges"] <= r["pymetis_inter_edges"]
    )
    print(
        f"QA cut <= PyMetis cut: {wins}/{len(feasible)} "
        f"({100 * wins / len(feasible):.1f}%)"
    )

    diffs = [r["pymetis_inter_edges"] - r["qa_inter_edges"] for r in feasible]
    print(f"Mean absolute difference vs PyMetis: {statistics.mean(diffs):.1f}")

    pct = [
        100 * (r["pymetis_inter_edges"] - r["qa_inter_edges"]) / r["qa_inter_edges"]
        for r in feasible
        if r["qa_inter_edges"] > 0
    ]
    if pct:
        print(f"Mean percentage difference vs PyMetis: {statistics.mean(pct):.3f}%")

    print(
        "\nNote: the paired Wilcoxon signed-rank test reported in Section 6.1 "
        "is computed separately from the stored rows; see sql/comparision_of_succes.sql."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["hybrid", "qpu"],
        default="hybrid",
        help="Solver to evaluate (default: hybrid).",
    )
    parser.add_argument(
        "--scope",
        type=int,
        default=126,
        help="Number of graphs to evaluate (paper used 126 hybrid, 70 QPU).",
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
    evaluate(scope=args.scope, mode=args.mode, seed=args.seed)
