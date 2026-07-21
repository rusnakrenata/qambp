import math
import os
import time
import numpy as np
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridSampler
from qubo_matrix import generate_qubo_matrix

# Optional Ocean SDK profile name (a section in ~/.config/dwave/dwave.conf).
# If unset, Ocean falls back to its own default resolution order, which
# includes the DWAVE_API_TOKEN environment variable.
DWAVE_PROFILE = os.getenv("DWAVE_PROFILE") or None

# Sentinel cut value recorded when a solver returns no balanced sample.
# The analysis queries treat any row with qa_inter_edges >= 1000000 as
# infeasible, so this value must stay at or above that threshold.
INVALID_CUT = int(1e6)

def qa_testing(G, lambda_par, comp_type='hybrid', num_reads=10):
    """
    Run QUBO formulation of the graph partitioning problem using a specified sampler.

    Parameters:
        G: NetworkX graph
        lambda_par: Penalty coefficient
        comp_type: 'test', 'hybrid', or 'qpu'
        num_reads: Number of reads for classical or QPU runs

    Returns:
        qa_inter_edges: Number of edges crossing between partitions
        best_partition: List of 0/1 indicating node assignments
        duration_qa: Runtime of hybrid sampling
    """
    duration_qa = 0.0

    # Construct the QUBO matrix
    Q = generate_qubo_matrix(G, lambda_par)

    if comp_type == 'test':
        # Classical simulated annealing
        sa_sampler = SimulatedAnnealingSampler()
        response = sa_sampler.sample_qubo(Q, num_reads=num_reads)

    elif comp_type == 'hybrid':
        # Use hybrid solver via Leap
        bqm = BinaryQuadraticModel.from_qubo(Q)
        start_time = time.time()
        hybrid_sampler = LeapHybridSampler(profile=DWAVE_PROFILE)
        response = hybrid_sampler.sample(bqm)
        duration_qa = time.time() - start_time
                

    elif comp_type == 'qpu':
        # Use QPU solver. EmbeddingComposite applies the minorminer heuristic;
        # default annealing time (20 us) and chain strength are used.
        start_time = time.time()
        qpu_sampler = EmbeddingComposite(DWaveSampler(profile=DWAVE_PROFILE))
        response = qpu_sampler.sample_qubo(Q, num_reads=num_reads)
        duration_qa = time.time() - start_time

    else:
        raise ValueError(
            f"Unknown comp_type {comp_type!r}; expected 'test', 'hybrid' or 'qpu'."
        )

    n_nodes = G.number_of_nodes()
    balanced_sizes = (math.floor(n_nodes / 2), math.ceil(n_nodes / 2))

    best_cut = None
    best_partition = None

    for sample in response.record:
        partition = list(map(int, sample[0]))

        # Ensure balanced split (±1 node)
        if np.sum(partition) not in balanced_sizes:
            continue

        qa_inter_edges = sum(
            partition[u] + partition[v] - 2 * partition[u] * partition[v]
            for u, v in G.edges()
        )
        if best_cut is None or qa_inter_edges < best_cut:
            best_cut = qa_inter_edges
            best_partition = partition

    del response

    if best_partition is None:
        # No feasible (balanced) sample was returned. INVALID_CUT is the
        # sentinel the analysis queries filter on (qa_inter_edges < 1000000).
        return INVALID_CUT, [], duration_qa

    return best_cut, list(best_partition), duration_qa
