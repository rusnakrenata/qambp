from database_tables import Graph, Partition
from qa_testing import qa_testing
import datetime, time, json, uuid
import numpy as np


def store_graph_to_db(session, graph_data, pymetis_data, kl_data, i='test', description="No description"):
    """
    Store graph properties into the database.

    Parameters:
        session: SQLAlchemy DB session
        graph_data: Dict of graph metrics (from generate_graph)
        i: Suffix index or identifier
        description: Description for the graph

    Returns:
        Graph SQLAlchemy model instance
    """
    timestamp = datetime.datetime.now()

    # `name` is UNIQUE. Second resolution alone collides whenever two graphs
    # are generated within the same second, which happens routinely for the
    # small graphs used in the QPU experiments, so include microseconds and a
    # short random suffix.
    stamp = timestamp.strftime('%Y%m%d_%H%M%S_%f')
    suffix = uuid.uuid4().hex[:6]

    new_graph = Graph(
        name=f"Graph_{stamp}_{i}_{suffix}",
        description=description,
        nr_of_nodes=graph_data["nr_of_nodes"],
        edge_probability=graph_data["edge_probability"],
        edges=graph_data["edges"],
        pymetis_partition_nodes=pymetis_data["pymetis_partition"],
        pymetis_inter_edges=pymetis_data["pymetis_inter_edges"],
        pymetis_intra_edges=pymetis_data["pymetis_intra_edges"],
        lambda_est=graph_data["lambda_est"],
        density=graph_data["density"],
        clustering=graph_data["clustering"],
        nr_of_edges=graph_data["nr_of_edges"],
        kernighan_lin_inter_edges=kl_data["kl_inter_edges"],
        duration_keringham_lin=kl_data["duration_keringham_lin"],
        # NOTE: the PyMetis runtime is produced by run_pymetis_partition(),
        # not by generate_graph().
        duration_pymetis=pymetis_data["duration_pymetis"]
    )

    session.add(new_graph)
    session.commit()

    return new_graph


def store_partition_to_db(session, G, db_graph, lambda_total, comp_type, num_reads,
                          lambda_mult=None):
    """
    Run the partitioning algorithm for a single graph and store the results in the database.

    Parameters:
        session: Database session
        G: NetworkX graph
        lambda_total: Penalty parameter actually used in the QUBO
                      (lambda_est * lambda_mult, see Eq. 27)
        comp_type: Type of computation to perform ('hybrid', 'qpu' or 'test')
        num_reads: Number of reads for the QPU / classical sampler
        lambda_mult: The multiplier alone, stored so the calibration
                     queries do not have to recover it by division.

    Returns:
        The committed Partition instance.
    """
    start_time = time.time()

    inter_edges, partition, duration_qa = qa_testing(
        G, lambda_total, comp_type=comp_type, num_reads=num_reads
    )
    end_time = time.time()
    duration_full =end_time - start_time

    # Get the last partition's id
    last_partition = session.query(Partition).order_by(Partition.id.desc()).first()

    # If a partition exists, increment the last ID by 1
    if last_partition:
        new_id = last_partition.id + 1
    else:
        # If no partitions exist, start from 1
        new_id = 1

    # Create a new partition instance
    new_partition = Partition(
        id=int(new_id),
        graph_id=int(db_graph.id),
        lambda_total=float(lambda_total),
        lambda_mult=float(lambda_mult) if lambda_mult is not None else None,
        comp_type=comp_type,
        qa_inter_edges=int(inter_edges),
        nodes=json.dumps(np.array(partition, dtype=np.int8).astype(int).tolist()),
        duration_full=duration_full,
        duration_qa=duration_qa
    )

    # Add the partition to the session
    session.add(new_partition)
    session.commit()

    return new_partition
