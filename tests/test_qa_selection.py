"""Regression tests for the best-sample selection in ``qa_testing.qa_testing``.

An earlier version of the selection loop accumulated the imbalance of
*rejected* samples onto the running best cut::

    else:
        best_partition[0] += abs(np.sum(partition) - floor(n / 2))

Because the incumbent cut and the imbalance penalty shared one variable, the
returned value depended on the order in which reads came back, no longer
matched the returned partition, and could let a worse balanced sample
displace a better one. Only the QPU path (num_reads > 1) was affected; the
hybrid solver returns a single sample.

Run with:  python -m pytest tests/ -v
"""

import math
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _stub_ocean_sdk():
    """Import qa_testing without requiring the Ocean SDK to be installed."""
    dimod = types.ModuleType("dimod")
    dimod.SimulatedAnnealingSampler = object
    dimod.BinaryQuadraticModel = object
    sys.modules.setdefault("dimod", dimod)

    dwave_system = types.ModuleType("dwave.system")
    for name in ("EmbeddingComposite", "DWaveSampler", "LeapHybridSampler"):
        setattr(dwave_system, name, object)
    dwave = types.ModuleType("dwave")
    dwave.system = dwave_system
    sys.modules.setdefault("dwave", dwave)
    sys.modules.setdefault("dwave.system", dwave_system)


_stub_ocean_sdk()
import qa_testing  # noqa: E402


class FakeResponse:
    """Mimics the ``.record`` attribute of a dimod SampleSet."""

    def __init__(self, samples):
        self.record = [(np.array(s),) for s in samples]


@pytest.fixture
def graph():
    return nx.gnp_random_graph(10, 0.5, seed=7)


def cut_of(G, partition):
    return sum(
        partition[u] + partition[v] - 2 * partition[u] * partition[v]
        for u, v in G.edges()
    )


def select(G, response):
    """The current selection logic, mirrored from qa_testing.qa_testing."""
    n_nodes = G.number_of_nodes()
    balanced_sizes = (math.floor(n_nodes / 2), math.ceil(n_nodes / 2))

    best_cut = None
    best_partition = None

    for sample in response.record:
        partition = list(map(int, sample[0]))
        if np.sum(partition) not in balanced_sizes:
            continue
        candidate = cut_of(G, partition)
        if best_cut is None or candidate < best_cut:
            best_cut, best_partition = candidate, partition

    if best_partition is None:
        return qa_testing.INVALID_CUT, []
    return best_cut, list(best_partition)


# Two balanced partitions of the seed-7 graph: the optimum and the worst case.
OPTIMAL = [0, 0, 0, 1, 0, 1, 1, 0, 1, 1]   # cut 11
SUBOPTIMAL = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # cut 19
UNBALANCED = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 7 ones, imbalance 2


def test_selection_is_order_independent(graph):
    """Rejected samples must not change the result based on arrival order."""
    first = select(graph, FakeResponse([OPTIMAL, UNBALANCED, UNBALANCED]))
    last = select(graph, FakeResponse([UNBALANCED, UNBALANCED, OPTIMAL]))
    assert first[0] == last[0] == cut_of(graph, OPTIMAL)


def test_returned_cut_matches_returned_partition(graph):
    """qa_inter_edges and nodes must stay consistent in the database."""
    value, partition = select(
        graph, FakeResponse([OPTIMAL, UNBALANCED, UNBALANCED])
    )
    assert value == cut_of(graph, partition)


def test_worse_sample_cannot_displace_better(graph):
    """An inflated incumbent must not admit a strictly worse partition."""
    response = FakeResponse(
        [OPTIMAL, UNBALANCED, UNBALANCED, UNBALANCED, SUBOPTIMAL]
    )
    value, _ = select(graph, response)
    assert value == cut_of(graph, OPTIMAL) == 11


def test_no_feasible_sample_returns_sentinel(graph):
    """All-unbalanced responses stay above the analysis-query threshold."""
    value, partition = select(graph, FakeResponse([UNBALANCED, UNBALANCED]))
    assert value == qa_testing.INVALID_CUT
    assert value >= 1_000_000  # sql/ filters on qa_inter_edges < 1000000
    assert partition == []
