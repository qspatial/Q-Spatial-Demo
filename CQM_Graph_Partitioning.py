from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler
import networkx as nx


def create_spatial_graph(grid_size=5):

    G = nx.grid_2d_graph(grid_size, grid_size)

    G = nx.convert_node_labels_to_integers(G)

    return G


def solve_graph_partitioning_cqm(graph, num_partitions=3, min_partition_size=None, time_limit=5):

    nodes = list(graph.nodes())
    edges = list(graph.edges())
    partitions = list(range(num_partitions))

    # Set minimum partition size
    s = min_partition_size if min_partition_size else len(nodes) // num_partitions

    # Create binary variables
    x = [[Binary(f'x_{{{i},{l}}}') for l in partitions] for i in nodes]

    # Create CQM
    cqm = ConstrainedQuadraticModel()

    # Build objective function
    min_obj_funs = []
    for i, j in edges:
        for l in partitions:
            min_obj_funs.append(graph[i][j].get('weight', 1.0) *
                                (x[i][l] + x[j][l] - 2 * x[i][l] * x[j][l]))

    cqm.set_objective(sum(min_obj_funs))

    # Add one-hot constraints for each node
    for i in nodes:
        cqm.add_discrete([f'x_{{{i},{l}}}' for l in partitions],
                         label=f'one-hot-node-{i}')

    # Add partition size constraints
    for l in partitions:
        cqm.add_constraint(
            sum(x[i][l] for i in range(len(nodes))) >= s,
            label=f'partition-size-{l}'
        )

    # Submit to D-Wave solver
    print("Submitting CQM to D-Wave solver...")
    print(f"  Time limit: {time_limit} seconds")
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm,
                                   label='Graph Partitioning - CQM',
                                   time_limit=time_limit)

    best_solution = sampleset.first

    print(f"\nResults:")
    print(f"  Objective value (edge cuts): {int(best_solution.energy)}")
    print(f"  Feasible: {best_solution.is_feasible}")

    # Convert solution to node-partition mapping
    node_assignment = {}
    for i in nodes:
        for l in partitions:
            if best_solution.sample.get(f'x_{{{i},{l}}}', 0) == 1:
                node_assignment[i] = l
                break

    return node_assignment, best_solution


def visualize_partition(graph, node_assignment):
    """Print partition assignments"""
    print("\nPartition Assignments:")
    partitions = {}
    for node, partition in node_assignment.items():
        if partition not in partitions:
            partitions[partition] = []
        partitions[partition].append(node)

    for partition, nodes in sorted(partitions.items()):
        print(f"  Partition {partition}: {sorted(nodes)} ({len(nodes)} nodes)")

    edge_cuts = 0
    for i, j in graph.edges():
        if node_assignment[i] != node_assignment[j]:
            edge_cuts += 1

    print(f"\n  Total edge cuts: {edge_cuts}")
    print(f"  Total nodes assigned: {len(node_assignment)}/{len(graph.nodes())}")


if __name__ == "__main__":
    # Create spatial graph
    print("CQM Solver - Spatial Graph Partitioning")



    print("\nCreating spatial graph...")
    G = create_spatial_graph(grid_size=4)  # 4x4 = 16 nodes
    print(f"Graph created: {len(G.nodes())} nodes, {len(G.edges())} edges\n")


    num_partitions = 4
    solution, result = solve_graph_partitioning_cqm(G,
                                                     num_partitions=num_partitions,
                                                     time_limit=15)  # Give solver more time


    visualize_partition(G, solution)
