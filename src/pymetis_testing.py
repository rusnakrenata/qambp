import pymetis
import time

def run_pymetis_partition(G):
    """
    Run METIS partitioning on a graph and return stats.

    Parameters:
        G: NetworkX graph

    Returns:
        dict: METIS partition info
    """
    try:
        start = time.time()
        _, partition = pymetis.part_graph(2, adjacency=G)
        duration = time.time() - start

        # PyMetis >= 2025.x returns an array.array, which is not JSON
        # serializable and would break the JSON column in `graphs`.
        partition = [int(x) for x in partition]

        intra_edges = sum(1 for u, v in G.edges() if partition[u] == partition[v])
        inter_edges = sum(1 for u, v in G.edges() if partition[u] != partition[v])
    except Exception as exc:
        print(f"PyMetis partitioning failed: {exc}")
        partition = []
        intra_edges = inter_edges = 0
        duration = 0.0

    return {
        "pymetis_partition": partition,
        "pymetis_intra_edges": intra_edges,
        "pymetis_inter_edges": inter_edges,
        "duration_pymetis": duration
    }
